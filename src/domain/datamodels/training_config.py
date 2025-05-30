from pydantic import BaseModel, Field, model_validator, field_validator
from src.utils.helper_functions import get_project_root
from typing import Optional
import os


class TrainingConfig(BaseModel):
    num_epochs: int = Field(default=2)
    batch_size_train: int = Field(default=32)
    batch_size_val: int = Field(default=32)
    train_test_split: float = Field(default=0.8)
    max_nb_steps: Optional[int] = Field(default=None)
    learning_rate: float = Field(default=0.001)
    training_pathway: str = Field(default="o2p")
    save_every: int = Field(default=1)
    model_artifacts_dir: str = Field(default="model_artifacts")
    weight_decay: float = Field(default=0.0)
    checkpoint_path: Optional[str] = Field(default=None)
    test_data_path: Optional[str] = Field(default=None)
    num_chunks: Optional[int] = Field(
        default=1,
        description="Number of chunks to split a batch into for accumulated gradients",
    )

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
        # Create the directory if it doesn't exist
        os.makedirs(values["model_artifacts_dir"], exist_ok=True)
        if "test_data_path" in values and values["test_data_path"]:
            values["test_data_path"] = os.path.join(
                project_root, "data", values["test_data_path"]
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

    @model_validator(mode="after")
    def validate_paths(self):
        if not os.path.exists(self.model_artifacts_dir):
            raise FileNotFoundError(
                f"Model directory not found: {self.model_artifacts_dir}"
            )
        if self.checkpoint_path and not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {self.checkpoint_path}"
            )
        return self
