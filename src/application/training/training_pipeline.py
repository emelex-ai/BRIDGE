import os
import random
import torch
from tqdm import tqdm
from typing import Dict, Any
from src.application.training.ortho_metrics import calculate_orth_metrics
from src.application.training.phon_metrics import calculate_phon_metrics
from src.domain.datamodels import TrainingConfig
from src.domain.dataset import BridgeDataset
from src.domain.model import Model
from typing import Dict, Union
import time
from torch.profiler import profile, record_function, ProfilerActivity
import wandb
from traindata import utilities


class TrainingPipeline:
    def __init__(
        self, model: Model, training_config: TrainingConfig, dataset: BridgeDataset
    ):
        self.training_config = training_config
        self.dataset = dataset
        self.device = torch.device(training_config.device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        self.train_slices, self.val_slices = self.create_data_slices()
        self.phon_reps = torch.tensor(
            utilities.phontable("data/phonreps.csv").values,
            dtype=torch.float,
            device=self.device,
        )[:-1]
        self.start_epoch = 0
        if training_config.checkpoint_path:
            self.load_model(training_config.checkpoint_path)

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
            phon_loss = torch.nn.CrossEntropyLoss(ignore_index=2)(
                logits["phon"], phonology["targets"]
            )

        # Calculate orth_loss if applicable
        if self.training_config.training_pathway in ["p2o", "op2op"]:
            orth_loss = torch.nn.CrossEntropyLoss(ignore_index=4)(
                logits["orth"], orthography["enc_input_ids"][:, 1:]
            )

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

    def single_step(self, batch_slice: slice, calculate_metrics: bool = False) -> dict:
        batch = self.dataset[batch_slice]
        orthography, phonology = batch["orthographic"], batch["phonological"]

        # Forward pass
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        #     with record_function("model_forward"):
        logits = self.forward(orthography, phonology)
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

        # Compute loss
        metrics = self.compute_loss(logits, orthography, phonology)

        # Backpropagation only if training
        if self.model.training:
            metrics.get("loss").backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        if calculate_metrics:
            metrics.update(self.compute_metrics(logits, orthography, phonology))
        return metrics

    def train_single_epoch(self, epoch: int) -> dict:
        self.model.train()
        start = time.time()
        cutpoint = int(len(self.dataset) * self.training_config.train_test_split)
        self.dataset.shuffle(cutpoint)
        progress_bar = tqdm(self.train_slices, desc=f"Training Epoch {epoch+1}")
        total_metrics = {}
        for step, batch_slice in enumerate(progress_bar):
            metrics = self.single_step(batch_slice, False)
            progress_bar.set_postfix(
                {key: f"{value:.4f}" for key, value in metrics.items()}
            )
            if not total_metrics:
                total_metrics = metrics
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
                metrics = self.single_step(batch_slice, True)
                progress_bar.set_postfix(
                    {key: f"{value:.4f}" for key, value in metrics.items()}
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

    def run_train_val_loop(self, run_name: str):
        for epoch in range(self.start_epoch, self.training_config.num_epochs):
            training_metrics = self.train_single_epoch(epoch)
            if self.val_slices:
                metrics = self.validate_single_epoch(epoch)
                training_metrics.update(metrics)
            self.save_model(epoch, run_name)
            yield training_metrics

    def save_model(self, epoch: int, run_name: str) -> None:
        if (epoch + 1) % self.training_config.save_every == 0:

            model_path = (
                f"{self.training_config.model_artifacts_dir}/model_epoch_{epoch+1}.pth"
            )
            torch.save(
                {
                    "model_config": self.model.model_config,
                    "dataset_config": self.model.dataset_config,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": epoch,
                },
                model_path,
            )

    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"]
