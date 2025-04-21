from pydantic import BaseModel, Field, field_validator, ValidationInfo, model_validator
from src.utils.helper_functions import get_project_root
from typing import List, Optional
import os

from pydantic import ConfigDict


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    num_phon_enc_layers: int = Field(default=2)
    num_orth_enc_layers: int = Field(default=2)
    num_mixing_enc_layers: int = Field(default=2)
    num_phon_dec_layers: int = Field(default=2)
    num_orth_dec_layers: int = Field(default=2)
    d_model: int = Field(default=64)
    nhead: int = Field(default=2)
    d_embedding: int = Field(default=1)
    seed: Optional[int] = Field(default=None)

    @field_validator("d_model")
    def validate_d_model(cls, v, info: ValidationInfo):
        nhead = info.data.get("nhead")
        if nhead is not None and v % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        return v
