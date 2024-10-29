from pydantic import BaseModel, Field, field_validator, ValidationInfo, model_validator, PositiveInt
from src.utils.helper_funtions import get_project_root
from typing import List, Optional
import os


class TrainingConfig(BaseModel):
    device: str = Field(default="cpu")
    num_epochs: int = Field(default=2)
    batch_size_train: int = Field(default=32)
    batch_size_val: int = Field(default=32)
    train_test_split: float = Field(default=0.8)
    max_nb_steps: Optional[int] = Field(default=None)
    learning_rate: float = Field(default=0.001)
    save_every: int = Field(default=1)
    model_artifacts_dir: str = Field(default="models")
    model_id: Optional[str] = Field(default=None)

    @model_validator(mode="before")
    def convert_paths(cls, values):
        """Convert relative paths to absolute paths before validation occurs."""
        project_root = get_project_root()

        values.setdefault("model_artifacts_dir", cls.model_fields["model_artifacts_dir"].get_default())
        values["model_artifacts_dir"] = os.path.join(project_root, values["model_artifacts_dir"])

        return values

    @model_validator(mode="after")
    def validate_paths(self):
        if not os.path.exists(self.model_artifacts_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_artifacts_dir}")

        return self

    class Config:
        protected_namespaces = ()
