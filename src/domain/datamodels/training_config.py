from pydantic import BaseModel, Field, model_validator, field_validator
from src.utils.helper_funtions import get_project_root
from src.utils.device import device_manager
from typing import Optional
import os


class TrainingConfig(BaseModel):
    device: Optional[str] = Field(default=device_manager.device.type)
    num_epochs: int = Field(default=2, description="Number of epochs to train the model.")
    batch_size_train: int = Field(default=32, description="Batch size for training.")
    batch_size_val: int = Field(default=32, description="Batch size for validation.")
    train_test_split: float = Field(default=0.8, description="Fraction of data to use for training.")
    learning_rate: float = Field(default=0.001, description="Learning rate for the optimizer.")
    training_pathway: str = Field(default="o2p", description="Training pathway: o2p, p2o, op2op, p2p.")
    save_every: int = Field(default=1, description="Save model every n epochs.")
    model_artifacts_dir: str = Field(default="model_artifacts", description="Directory to save model artifacts.")
    weight_decay: float = Field(default=0.0, description="Weight decay for the optimizer.")


    @model_validator(mode="before")
    def convert_paths(cls, values):
        """Convert relative paths to absolute paths before validation occurs."""
        project_root = get_project_root()
        values.setdefault(
            "model_artifacts_dir", cls.model_fields["model_artifacts_dir"].get_default()
        )
        values["model_artifacts_dir"] = os.path.join(
            project_root, values["model_artifacts_dir"]
        )
        return values

    @field_validator("training_pathway")
    def validate_pathway(cls, v: str) -> str:
        allowed_training_pathways = ["o2p", "p2o", "op2op", "p2p"]
        if v not in allowed_training_pathways:
            raise ValueError(
                f"Invalid pathway: {v}. Allowed: {allowed_training_pathways}"
            )
        return v

    @field_validator("train_test_split")
    def validate_train_test_split(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("train_test_split must be between 0.0 and 1.0")
        return v

    @field_validator("device", mode="before")
    def validate_device(cls, v: str | None) -> str:
        if v is None:
            v = device_manager.device.type
        return v

    @model_validator(mode="after")
    def validate_paths(self):
        if not os.path.exists(self.model_artifacts_dir):
            raise FileNotFoundError(
                f"Model directory not found: {self.model_artifacts_dir}"
            )
        return self

    class Config:
        protected_namespaces = ()
