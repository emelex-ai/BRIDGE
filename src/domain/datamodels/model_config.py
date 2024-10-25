from pydantic import BaseModel, Field, field_validator, ValidationInfo, model_validator
from src.utils.helper_funtions import get_project_root
from typing import List, Optional
import os


class ModelConfig(BaseModel):
    device: str = Field(default="cpu")
    project: str = Field(default="Bridge")
    num_epochs: int = Field(default=2)
    batch_size_train: int = Field(default=32)
    batch_size_val: int = Field(default=32)
    num_phon_enc_layers: int = Field(default=2)
    num_orth_enc_layers: int = Field(default=2)
    num_mixing_enc_layers: int = Field(default=2)
    num_phon_dec_layers: int = Field(default=2)
    num_orth_dec_layers: int = Field(default=2)
    learning_rate: float = Field(default=0.001)
    d_model: int = Field(default=64)
    nhead: int = Field(default=2)
    train_test_split: float = Field(default=0.8)
    d_embedding: int = Field(default=1)
    seed: int = Field(default=1337)
    model_artifacts_dir: str = Field(default="models")
    model_id: Optional[str] = Field(default=None)
    pathway: str = Field(default="o2p")
    save_every: int = Field(default=1)
    max_nb_steps: int = Field(default=10)
    test_filenames: Optional[List[str]] = Field(default=None)

    @model_validator(mode="before")
    def convert_paths(cls, values):
        """Convert relative paths to absolute paths before validation occurs."""
        project_root = get_project_root()

        values.setdefault("model_artifacts_dir", cls.model_fields["model_artifacts_dir"].get_default())
        values.setdefault("test_filenames", cls.model_fields["test_filenames"].get_default())

        values["model_artifacts_dir"] = os.path.join(project_root, values["model_artifacts_dir"])

        if values.get("test_filenames"):
            values["test_filenames"] = [
                os.path.join(project_root, "data", "tests", filename) for filename in values["test_filenames"]
            ]

        return values

    @field_validator("d_model")
    def validate_d_model(cls, v, info: ValidationInfo):
        nhead = info.data.get("nhead")
        if nhead is not None and v % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

    @field_validator("pathway")
    def validate_pathway(cls, v):
        allowed_pathways = ["o2p", "p2o", "op2op"]
        if v not in allowed_pathways:
            raise ValueError(f"Invalid pathway: {v}. Allowed: {allowed_pathways}")

    @model_validator(mode="after")
    def validate_paths(self):
        if not os.path.exists(self.model_artifacts_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_artifacts_dir}")

        if self.test_filenames:
            for filename in self.test_filenames:
                if not os.path.exists(filename):
                    raise FileNotFoundError(f"Test file not found: {filename}")

        return self

    class Config:
        protected_namespaces = ()
