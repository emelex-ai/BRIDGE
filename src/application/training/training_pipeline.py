import json
import os
import gc
from typing import Union
import time

import torch
from tqdm import tqdm

from src.application.training.ortho_metrics import calculate_orth_metrics
from src.application.training.phon_metrics import calculate_phon_metrics
from src.domain.datamodels import TrainingConfig
from src.domain.dataset import BridgeDataset
from src.domain.model import Model
from src.utils.device_manager import device_manager
from src.infra.metrics.metrics_logger import MetricsLogger


class TrainingPipeline:

    def __init__(
        self,
        model: Model,
        training_config: TrainingConfig,
        dataset: BridgeDataset,
        metrics_logger: MetricsLogger,
    ):
        self.training_config = training_config
        self.dataset = dataset
        self.test_dataset = None
        if self.training_config.test_data_path:
            test_dataset_config = self.dataset.dataset_config.model_copy()
            test_dataset_config.dataset_filepath = self.training_config.test_data_path
            self.test_dataset = BridgeDataset(
                dataset_config=test_dataset_config, gcs_client=self.dataset.gcs_client
            )
        self.device = device_manager.device
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        self.train_slices, self.val_slices = self.create_data_slices()
        self.phon_reps = self.dataset.tokenizer.phoneme_tokenizer.phonreps_array

        self.start_epoch = 0
        if training_config.checkpoint_path:
            self.load_model(training_config.checkpoint_path)
        self.metrics_logger = metrics_logger

    def create_data_slices(self):
        cutpoint = int(len(self.dataset) * self.training_config.train_test_split)
        train_slices = [
            slice(i, min(i + self.training_config.batch_size_train, cutpoint))
            for i in range(0, cutpoint, self.training_config.batch_size_train)
        ]
        val_slices = [
            slice(i, min(i + self.training_config.batch_size_val, len(self.dataset)))
            for i in range(
                cutpoint, len(self.dataset), self.training_config.batch_size_val
            )
        ]
        return train_slices, val_slices

    def forward(
        self, orthography: dict[str, torch.Tensor], phonology: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if self.training_config.training_pathway == "o2p":
            return self.model(
                task="o2p",
                orth_enc_input=orthography["enc_input_ids"],
                orth_enc_pad_mask=orthography["enc_pad_mask"],
                phon_dec_input=phonology["dec_input_ids"],
                phon_dec_pad_mask=phonology["dec_pad_mask"],
            )
        elif self.training_config.training_pathway == "op2op":
            return self.model(
                task="op2op",
                orth_enc_input=orthography["enc_input_ids"],
                orth_enc_pad_mask=orthography["enc_pad_mask"],
                orth_dec_input=orthography["dec_input_ids"],
                orth_dec_pad_mask=orthography["dec_pad_mask"],
                phon_enc_input=phonology["enc_input_ids"],
                phon_enc_pad_mask=phonology["enc_pad_mask"],
                phon_dec_input=phonology["dec_input_ids"],
                phon_dec_pad_mask=phonology["dec_pad_mask"],
            )
        elif self.training_config.training_pathway == "p2o":
            return self.model(
                task="p2o",
                phon_enc_input=phonology["enc_input_ids"],
                phon_enc_pad_mask=phonology["enc_pad_mask"],
                orth_dec_input=orthography["dec_input_ids"],
                orth_dec_pad_mask=orthography["dec_pad_mask"],
            )
        elif self.training_config.training_pathway == "p2p":
            return self.model(
                task="p2p",
                phon_enc_input=phonology["enc_input_ids"],
                phon_enc_pad_mask=phonology["enc_pad_mask"],
                phon_dec_input=phonology["dec_input_ids"],
                phon_dec_pad_mask=phonology["dec_pad_mask"],
            )

    def compute_loss(
        self,
        logits: dict[str, torch.Tensor],
        orthography: dict[str, torch.Tensor],
        phonology: dict[str, torch.Tensor],
    ) -> dict[str, Union[torch.Tensor, None]]:
        # Initialize losses to None
        orth_loss = None
        phon_loss = None

        # Calculate phon_loss if applicable
        if self.training_config.training_pathway in ["o2p", "op2op", "p2p"]:
            phon_loss = torch.nn.CrossEntropyLoss(ignore_index=35)(
                logits["phon"], phonology["targets"]
            )  # Ignore [PAD] token
            phon_loss = torch.nn.CrossEntropyLoss(ignore_index=35)(
                logits["phon"], phonology["targets"]
            )  # Ignore [PAD] token

        # Calculate orth_loss if applicable
        if self.training_config.training_pathway in ["p2o", "op2op"]:
            orth_loss = torch.nn.CrossEntropyLoss(ignore_index=2)(
                logits["orth"], orthography["enc_input_ids"][:, 1:]
            )  # Ignore [PAD] token
            orth_loss = torch.nn.CrossEntropyLoss(ignore_index=2)(
                logits["orth"], orthography["enc_input_ids"][:, 1:]
            )  # Ignore [PAD] token

        # Calculate the combined loss, summing only non-None losses
        total_loss = sum(loss for loss in [orth_loss, phon_loss] if loss is not None)

        # Return only calculated losses, omitting None values
        loss_dict = {"loss": total_loss}
        if orth_loss is not None:
            loss_dict["orth_loss"] = orth_loss
        if phon_loss is not None:
            loss_dict["phon_loss"] = phon_loss

        return loss_dict

    def compute_metrics(
        self,
        logits: dict[str, torch.Tensor],
        orthography: dict[str, torch.Tensor],
        phonology: dict[str, torch.Tensor],
    ) -> dict:
        metrics = {}
        if self.training_config.training_pathway in ["o2p", "op2op", "p2p"]:
            metrics.update(calculate_phon_metrics(logits, phonology, self.phon_reps))

        if self.training_config.training_pathway in ["op2op", "p2o"]:
            metrics.update(calculate_orth_metrics(logits, orthography))

        return metrics

    def single_step(
        self,
        dataset: BridgeDataset,
        batch_slice: slice,
        calculate_metrics: bool = False,
    ) -> dict:
        # Split batch into smaller chunks if it's too large
        accumulated_metrics = {}
        sub_slices = self._create_sub_slices(
            batch_slice, num_chunks=4
        )  # Divide into 4 smaller batches

        # Zero gradients once at the beginning
        self.optimizer.zero_grad()

        for sub_slice in sub_slices:
            batch = dataset[sub_slice]
            orthography, phonology = batch["orthographic"], batch["phonological"]

            # Forward pass
            logits = self.forward(orthography, phonology)

            # Compute loss with scaled factor
            metrics = self.compute_loss(logits, orthography, phonology)
            loss = metrics.get("loss") / len(
                sub_slices
            )  # Scale loss by number of chunks

            # Backward pass (accumulate gradients)
            if self.model.training:
                loss.backward()

            # Update metrics
            if not accumulated_metrics:
                accumulated_metrics = {k: v for k, v in metrics.items()}
            else:
                for k, v in metrics.items():
                    if isinstance(v, torch.Tensor) and k != "word":
                        accumulated_metrics[k] += v

        # Only step optimizer after processing all sub-batches
        if self.model.training:
            self.optimizer.step()
            self.optimizer.zero_grad()

        if calculate_metrics:
            accumulated_metrics.update(
                self.compute_metrics(logits, orthography, phonology)
            )

        if self.metrics_logger.metrics_config.batch_metrics:
            self.metrics_logger.log_metrics(accumulated_metrics, "BATCH")

        accumulated_metrics.update({"word": json.dumps(dataset.words[batch_slice])})
        return accumulated_metrics

    def _create_sub_slices(self, batch_slice: slice, num_chunks: int) -> list[slice]:
        """Split a slice into smaller slices."""
        start, stop = batch_slice.start, batch_slice.stop
        size = stop - start
        chunk_size = max(1, size // num_chunks)

        sub_slices = []
        for i in range(0, size, chunk_size):
            sub_start = start + i
            sub_stop = min(start + i + chunk_size, stop)
            sub_slices.append(slice(sub_start, sub_stop))

        return sub_slices

    def train_single_epoch(self, epoch: int) -> dict:
        self.model.train()
        start = time.time()
        cutpoint = int(len(self.dataset) * self.training_config.train_test_split)
        # self.dataset.shuffle(cutpoint)
        progress_bar = tqdm(self.train_slices, desc=f"Training Epoch {epoch+1}")
        total_metrics = {}
        for step, batch_slice in enumerate(progress_bar):
            # Run garbage collection to free up memory
            if step % 10 == 0:
                gc.collect()

            metrics = self.single_step(
                self.dataset,
                batch_slice,
                self.metrics_logger.metrics_config.training_metrics,
            )
            # metrics = self.single_step(batch_slice, False)
            progress_bar.set_postfix(
                {
                    key: f"{value:.4f}"
                    for key, value in metrics.items()
                    if not isinstance(value, str)
                }
            )
            if not total_metrics:
                total_metrics = {
                    key: value
                    for key, value in metrics.items()
                    if not isinstance(value, str)
                }
            else:
                for key in total_metrics.keys():
                    total_metrics[key] += metrics[key]
        for key in total_metrics.keys():
            total_metrics[key] /= len(self.train_slices)
        total_metrics.update(
            {
                "time_per_step": (time.time() - start) / len(self.train_slices),
                "time_per_epoch": (time.time() - start) * len(self.train_slices),
            }
        )
        return {"train_" + str(key): val for key, val in total_metrics.items()}

    def validate_single_epoch(self, epoch: int) -> dict:
        self.model.eval()
        start = time.time()

        progress_bar = tqdm(self.val_slices, desc=f"Validating Epoch {epoch+1}")

        with torch.no_grad():
            total_metrics = {}
            for step, batch_slice in enumerate(progress_bar):
                metrics = self.single_step(
                    self.dataset,
                    batch_slice,
                    self.metrics_logger.metrics_config.validation_metrics,
                )
                progress_bar.set_postfix(
                    {
                        key: f"{value:.4f}"
                        for key, value in metrics.items()
                        if not isinstance(value, str)
                    }
                )
                if not total_metrics:
                    total_metrics = metrics
                else:
                    for key in total_metrics.keys():
                        total_metrics[key] += metrics[key]
            for key in total_metrics.keys():
                total_metrics[key] /= len(self.val_slices)
        total_metrics.update(
            {
                "time_per_step": (time.time() - start) / len(self.val_slices),
                "time_per_epoch": (time.time() - start) * len(self.val_slices),
            }
        )
        return {"valid_" + str(key): val for key, val in total_metrics.items()}

    def test_single_epoch(self, epoch: int) -> dict:
        self.model.eval()
        start = time.time()
        if self.test_dataset is None:
            raise ValueError("Test dataset not provided in the configuration.")

        # Create test slices based on batch size
        test_slices = [
            slice(
                i,
                min(i + self.training_config.batch_size_train, len(self.test_dataset)),
            )
            for i in range(
                0, len(self.test_dataset), self.training_config.batch_size_train
            )
        ]
        progress_bar = tqdm(test_slices, desc=f"Testing Epoch {epoch+1}")

        with torch.no_grad():
            total_metrics = {}
            for step, batch_slice in enumerate(progress_bar):
                metrics = self.single_step(
                    self.test_dataset,
                    batch_slice,
                    self.metrics_logger.metrics_config.validation_metrics,
                )
                progress_bar.set_postfix(
                    {
                        key: f"{value:.4f}"
                        for key, value in metrics.items()
                        if not isinstance(value, str)
                    }
                )
                if not total_metrics:
                    total_metrics = {
                        key: value
                        for key, value in metrics.items()
                        if not isinstance(value, str)
                    }
                else:
                    for key in total_metrics.keys():
                        total_metrics[key] += metrics[key]
            for key in total_metrics.keys():
                total_metrics[key] /= len(test_slices)
        total_metrics.update(
            {
                "time_per_step": (time.time() - start) / len(test_slices),
                "time_per_epoch": (time.time() - start) * len(test_slices),
            }
        )
        return {"test_" + str(key): val for key, val in total_metrics.items()}

    def run_train_val_loop(self, run_name: str):
        for epoch in range(self.start_epoch, self.training_config.num_epochs):
            training_metrics = self.train_single_epoch(epoch)
            if self.val_slices:
                metrics = self.validate_single_epoch(epoch)
                training_metrics.update(metrics)
            if self.test_dataset:
                metrics = self.test_single_epoch(epoch)
                training_metrics.update(metrics)
            self.metrics_logger.log_metrics(training_metrics, "EPOCH")
            self.save_model(epoch, run_name)
            yield training_metrics

    def save_model(self, epoch: int, run_name: str) -> None:
        if (epoch + 1) % self.training_config.save_every == 0:

            model_path = (
                f"{self.training_config.model_artifacts_dir}/model_epoch_{epoch}.pth"
            )
            torch.save(
                {
                    "model_config": self.model.model_config,
                    "dataset_config": self.dataset.dataset_config,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": epoch,
                },
                model_path,
            )
            if self.dataset.gcs_client:
                index = int(os.environ["CLOUD_RUN_TASK_INDEX"]) + 1
                self.dataset.gcs_client.upload_file(
                    os.environ["BUCKET_NAME"],
                    model_path,
                    f"pretraining/{index}/models/model_epoch_{epoch}.pth",
                )
            self.metrics_logger.save()

    def load_model(self, model_path: str):
        try:
            checkpoint = torch.load(model_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Set the correct starting epoch
            if "epoch" in checkpoint:
                self.start_epoch = checkpoint["epoch"] + 1  # Start from the next epoch
                logger.info(f"Resuming training from epoch {self.start_epoch}")
            else:
                logger.warning(
                    "Checkpoint doesn't contain epoch information, starting from 0"
                )
                self.start_epoch = 0

            return True
        except Exception as e:
            logger.error(f"Error loading checkpoint {model_path}: {e}")
            self.start_epoch = 0
            return False

    def transfer_partial_model_parameters(
        self, pretrained_model_path: str, module_prefixes: list[str]
    ):
        checkpoint = torch.load(pretrained_model_path)
        pretrained_state = checkpoint["model_state_dict"]
        filtered_state = {
            k: v
            for k, v in pretrained_state.items()
            if any(k.startswith(prefix) for prefix in module_prefixes)
        }

        model_dict = self.model.state_dict()
        model_dict.update(filtered_state)
        self.model.load_state_dict(model_dict)

        new_state = self.model.state_dict()
        for key, pretrained_weight in filtered_state.items():
            assert torch.equal(
                new_state[key], pretrained_weight
            ), f"Weight transfer failed for {key}"
