import torch as pt
from abc import ABC, abstractmethod
from tqdm import tqdm
import torch
from datetime import datetime
from src.domain.datamodels import TrainingConfig
from src.domain.dataset import ConnTextULDataset
from src.domain.model import Model
import time


class TrainingPipeline(ABC):

    def __init__(self, model: Model, training_config: TrainingConfig, dataset: ConnTextULDataset):
        self.training_config = training_config
        self.dataset = dataset
        self.device = pt.device(training_config.device)
        self.model = model.to(self.device)
        self.optimizer = pt.optim.AdamW(self.model.parameters(), lr=training_config.learning_rate)
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

    @abstractmethod
    def forward(self, orthography, phonology):
        """Abstract method to implement forward pass for each specific model"""
        pass

    @abstractmethod
    def compute_loss(self, logits, orthography, phonology):
        """Abstract method to compute loss specific to each model"""
        pass

    def single_step(self, batch_slice, epoch, step, example_ct):
        """Handles a single step of training or validation"""
        batch = self.dataset[batch_slice]
        orthography = batch["orthography"].to(self.device)
        phonology = batch["phonology"].to(self.device)

        # Forward pass
        logits = self.forward(orthography, phonology)

        # Compute loss
        loss = self.compute_loss(logits, orthography, phonology)

        # Backpropagation only if training
        if self.model.training:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Return metrics
        metrics = self.compute_metrics(logits, orthography, phonology, loss)
        return metrics

    @abstractmethod
    def compute_metrics(self, logits, orthography, phonology, loss):
        """Abstract method to compute custom metrics"""
        pass

    def train_single_epoch(self, epoch):
        self.model.train()
        example_ct = 0
        total_loss = 0

        start = time.time()

        # Determine the maximum steps for training
        max_steps = (
            min(self.training_config.max_nb_steps, len(self.train_slices))
            if self.training_config.max_nb_steps
            else len(self.train_slices)
        )
        progress_bar = tqdm(self.train_slices[:max_steps], desc=f"Training Epoch {epoch+1}", total=max_steps)

        for step, batch_slice in enumerate(progress_bar):
            metrics = self.single_step(batch_slice, epoch, step, example_ct)

            example_ct += len(self.dataset[batch_slice])
            total_loss += metrics["loss"]

            # Calculate and update time metrics
            metrics["time_per_train_step"] = (time.time() - start) / (step + 1)
            metrics["time_per_train_epoch"] = max_steps * metrics["time_per_train_step"]

            # Update progress bar with metrics
            progress_bar.set_postfix({key: f"{value:.4f}" for key, value in metrics.items()})

        return total_loss / max_steps

    def validate_single_epoch(self, epoch):
        self.model.eval()
        example_ct = 0
        total_loss = 0

        start = time.time()

        # Determine the maximum steps for validation
        max_steps = (
            min(self.training_config.max_nb_steps, len(self.val_slices))
            if self.training_config.max_nb_steps
            else len(self.val_slices)
        )
        progress_bar = tqdm(self.val_slices[:max_steps], desc=f"Validating Epoch {epoch+1}", total=max_steps)

        with torch.no_grad():
            for step, batch_slice in enumerate(progress_bar):
                metrics = self.single_step(batch_slice, epoch, step, example_ct)

                example_ct += len(self.dataset[batch_slice])
                total_loss += metrics["loss"]

                # Calculate and update time metrics
                metrics["time_per_val_step"] = (time.time() - start) / (step + 1)
                metrics["time_per_val_epoch"] = max_steps * metrics["time_per_val_step"]

                # Update progress bar with metrics
                progress_bar.set_postfix({key: f"{value:.4f}" for key, value in metrics.items()})

        return total_loss / max_steps

    def run_train_val_loop(self):
        for epoch in range(self.training_config.num_epochs):
            train_loss = self.train_single_epoch(epoch)
            if self.val_slices:
                val_loss = self.validate_single_epoch(epoch)
                print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}")
            self.save_model(epoch)

    def save_model(self, epoch):
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
    def set_seed(seed):
        """Set seed for reproducibility"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    @staticmethod
    def get_device(device=None):
        """Return the appropriate device (CPU or GPU)"""
        if device is None:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return device

    @staticmethod
    def get_model_file_name(model_id, epoch_num):
        return f"{model_id}_epoch_{epoch_num}.pth"

    @staticmethod
    def compare_state_dicts(state_dict1, state_dict2):
        """Compare two state dicts to check if they are identical"""
        if state_dict1.keys() != state_dict2.keys():
            return False
        return all(torch.allclose(state_dict1[key], state_dict2[key]) for key in state_dict1.keys())
