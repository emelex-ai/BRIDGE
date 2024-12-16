from pydantic import BaseModel, Field, field_validator, ValidationInfo, model_validator
from src.utils.helper_funtions import get_project_root
from typing import List, Optional
import os


class ModelConfig(BaseModel):
    num_phon_enc_layers: int = Field(default=2)
    num_orth_enc_layers: int = Field(default=2)
    num_mixing_enc_layers: int = Field(default=2)
    num_phon_dec_layers: int = Field(default=2)
    num_orth_dec_layers: int = Field(default=2)
    d_model: int = Field(default=64)
    nhead: int = Field(default=2)
    d_embedding: int = Field(default=1)
    seed: Optional[int] = Field(default=None)
    test_filenames: Optional[List[str]] = Field(default=None)

    @model_validator(mode="before")
    def convert_paths(cls, values):
        """Convert relative paths to absolute paths before validation occurs."""
        project_root = get_project_root()

        values.setdefault("test_filenames", cls.model_fields["test_filenames"].get_default())
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
        return v

    @model_validator(mode="after")
    def validate_paths(self):
        if self.test_filenames:
            for filename in self.test_filenames:
                if not os.path.exists(filename):
                    raise FileNotFoundError(f"Test file not found: {filename}")

        return self

    class Config:
        protected_namespaces = ()
