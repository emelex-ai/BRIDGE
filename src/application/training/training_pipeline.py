import torch
from tqdm import tqdm
from typing import Dict, Any
from src.domain.datamodels import TrainingConfig
from src.domain.dataset import ConnTextULDataset
from src.domain.model import Model
from typing import Dict, Union
import time
from torch.profiler import profile, record_function, ProfilerActivity


class TrainingPipeline:
    def __init__(self, model: Model, training_config: TrainingConfig, dataset: ConnTextULDataset):
        self.training_config = training_config
        self.dataset = dataset
        self.device = torch.device(training_config.device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=training_config.learning_rate)
        self.train_slices, self.val_slices = self.create_data_slices()

    def create_data_slices(self):
        cutpoint = int(len(self.dataset) * self.training_config.train_test_split)
        train_slices = [
            slice(i, min(i + self.training_config.batch_size_train, cutpoint))
            for i in range(0, cutpoint, self.training_config.batch_size_train)
        ]
        val_slices = [
            slice(i, min(i + self.training_config.batch_size_val, len(self.dataset)))
            for i in range(cutpoint, len(self.dataset), self.training_config.batch_size_val)
        ]
        return train_slices, val_slices

    def forward(
        self, orthography: Dict[str, torch.Tensor], phonology: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
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

    def compute_loss(
        self, logits: Dict[str, torch.Tensor], orthography: Dict[str, torch.Tensor], phonology: Dict[str, torch.Tensor]
    ) -> Dict[str, Union[torch.Tensor, None]]:
        # Initialize losses to None
        orth_loss = None
        phon_loss = None

        # Calculate phon_loss if applicable
        if self.training_config.training_pathway in ["o2p", "op2op"]:
            phon_loss = torch.nn.CrossEntropyLoss(ignore_index=2)(logits["phon"], phonology["targets"])

        # Calculate orth_loss if applicable
        if self.training_config.training_pathway in ["p2o", "op2op"]:
            orth_loss = torch.nn.CrossEntropyLoss(ignore_index=4)(logits["orth"], orthography["enc_input_ids"][:, 1:])

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
        logits: Dict[str, torch.Tensor],
        orthography: Dict[str, torch.Tensor],
        phonology: Dict[str, torch.Tensor],
    ) -> dict:

        if self.training_config.training_pathway in ["o2p", "op2op"]:
            phon_pred = torch.argmax(logits["phon"], dim=1)
            phon_true = phonology["targets"]
            phon_valid_mask = phon_true != 2
            masked_phon_true = phon_true[phon_valid_mask]
            masked_phon_pred = phon_pred[phon_valid_mask]
            correct_features = (masked_phon_pred == masked_phon_true).sum()
            phon_feature_accuracy = correct_features.float() / phon_valid_mask.sum().float()
            phoneme_wise_mask = phon_pred == phon_true
            phoneme_wise_accuracy = phoneme_wise_mask.all(dim=-1).sum() / (
                masked_phon_true.shape[0] / phon_true.shape[-1]
            )
            word_accuracies = [word[target != 2].all().int() for word, target in zip(phoneme_wise_mask, phon_true)]
            phon_word_accuracy = sum(word_accuracies) / len(word_accuracies)

            metrics = {
                "phon_feature_accuracy": phon_feature_accuracy.item(),
                "phon_phoneme_wise_accuracy": phoneme_wise_accuracy.item(),
                "phon_word_accuracy": phon_word_accuracy,
            }

        if self.training_config.training_pathway in ["op2op", "p2o"]:
            orth_pred = torch.argmax(logits["orth"], dim=1)
            orth_true = orthography["enc_input_ids"][:, 1:]
            orth_valid_mask = orth_true != 4
            masked_orth_true = orth_true[orth_valid_mask]
            masked_orth_pred = orth_pred[orth_valid_mask]
            correct_matches = (masked_orth_pred == masked_orth_true).sum()
            letter_wise_accuracy = correct_matches.float() / orth_valid_mask.sum().float()
            orth_pred[~orth_valid_mask] = 4
            word_wise_mask = orth_pred == orth_true
            orth_word_accuracy = word_wise_mask.all(dim=1).float().mean()

            metrics = {
                "letter_wise_accuracy": letter_wise_accuracy.item(),
                "word_wise_accuracy": orth_word_accuracy.item(),
            }

        return metrics

    def single_step(self, batch_slice: slice) -> Dict[str, Any]:
        batch = self.dataset[batch_slice]
        orthography, phonology = batch["orthography"], batch["phonology"]

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

        metrics.update(self.compute_metrics(logits, orthography, phonology))
        return metrics

    def train_single_epoch(self, epoch: int) -> float:
        self.model.train()
        start = time.time()
        progress_bar = tqdm(self.train_slices, desc=f"Training Epoch {epoch+1}")

        for step, batch_slice in enumerate(progress_bar):
            metrics = self.single_step(batch_slice)
            progress_bar.set_postfix({key: f"{value:.4f}" for key, value in metrics.items()})

        metrics.update(
            {
                "time_per_step": (time.time() - start) / len(self.train_slices),
                "time_per_epoch": (time.time() - start) * len(self.train_slices),
            }
        )
        return metrics

    def validate_single_epoch(self, epoch: int) -> float:
        self.model.eval()
        start = time.time()

        progress_bar = tqdm(self.val_slices, desc=f"Validating Epoch {epoch+1}")

        with torch.no_grad():
            for step, batch_slice in enumerate(progress_bar):
                metrics = self.single_step(batch_slice)
                progress_bar.set_postfix({key: f"{value:.4f}" for key, value in metrics.items()})

        metrics.update(
            {
                "time_per_step": (time.time() - start) / len(self.val_slices),
                "time_per_epoch": (time.time() - start) * len(self.val_slices),
            }
        )
        return metrics

    def run_train_val_loop(self) -> None:
        for epoch in range(self.training_config.num_epochs):
            metrices = self.train_single_epoch(epoch)
            if self.val_slices:
                metrices = self.validate_single_epoch(epoch)

            self.save_model(epoch)

    def save_model(self, epoch: int) -> None:
        if epoch % self.training_config.save_every == 0:
            model_path = f"{self.training_config.model_artifacts_dir}/model_epoch_{epoch}.pth"
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": epoch,
                },
                model_path,
            )

    @staticmethod
    def set_seed(seed: int) -> None:
        """Set seed for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
