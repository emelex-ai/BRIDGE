from src.domain.datamodels import ModelConfig, DatasetConfig
from src.domain.model.model import Model
from abc import ABC, abstractmethod
from typing import List, Dict
import torch as pt
import torch
import time


class TrainingPipeline(ABC):

    def __init__(self, model: Model, model_config: ModelConfig, dataset_config: DatasetConfig):
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.model = model
        self.opt = pt.optim.AdamW(model.parameters(), model_config.learning_rate)
        self.device = pt.device(model_config.device)
        self.model.to(self.device)
        self.epochs_completed = 0
        self.train_dataset_config_slices, self.val_dataset_config_slices = self.create_data_slices()

    @abstractmethod
    def forward(self, batch):
        """Forward pass logic specific to each model"""
        pass

    @abstractmethod
    def compute_loss(self, logits, batch):
        """Compute loss specific to each model"""
        pass

    def create_data_slices(self):
        cutpoint = len(self.dataset_config) * 0.8  # Assuming 80% train split
        train_slices = [
            slice(i, min(i + self.model_config.batch_size_train, cutpoint))
            for i in range(0, int(cutpoint), self.model_config.batch_size_train)
        ]
        val_slices = [
            slice(i, min(i + self.model_config.batch_size_val, len(self.dataset_config)))
            for i in range(int(cutpoint), len(self.dataset_config), self.model_config.batch_size_val)
        ]
        return train_slices, val_slices

    def train_single_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        example_ct = 0
        for step, batch_slice in enumerate(self.train_dataset_config_slices):
            batch = self.dataset_config[batch_slice]
            logits = self.forward(batch)
            loss = self.compute_loss(logits, batch)

            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

            total_loss += loss.item()
            example_ct += len(batch["orthography"])

        avg_loss = total_loss / len(self.train_dataset_config_slices)
        return {"train_loss": avg_loss, "epoch": epoch, "example_count": example_ct}

    def validate_single_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        example_ct = 0
        with torch.no_grad():
            for step, batch_slice in enumerate(self.val_dataset_config_slices):
                batch = self.dataset_config[batch_slice]
                logits = self.forward(batch)
                loss = self.compute_loss(logits, batch)

                total_loss += loss.item()
                example_ct += len(batch["orthography"])

        avg_loss = total_loss / len(self.val_dataset_config_slices)
        return {"val_loss": avg_loss, "epoch": epoch, "example_count": example_ct}

    def run_train_val_loop(self):
        for epoch in range(self.model_config.num_epochs):
            train_metrics = self.train_single_epoch(epoch)
            val_metrics = self.validate_single_epoch(epoch)
            combined_metrics = {**train_metrics, **val_metrics}
            self.log_metrics(combined_metrics)
            self.save_model(epoch)

    def log_metrics(self, metrics):
        print(f"Epoch {metrics['epoch']}: Train Loss {metrics['train_loss']}, Val Loss {metrics['val_loss']}")

    def save_model(self, epoch):
        if epoch % self.model_config.save_every == 0:
            model_path = f"{self.model_config.model_path}/model_epoch_{epoch}.pth"
            pt.save(self.model.state_dict(), model_path)
