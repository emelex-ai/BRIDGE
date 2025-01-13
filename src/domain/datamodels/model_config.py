from pydantic import BaseModel, Field, field_validator, ValidationInfo, model_validator, PositiveInt
from src.utils.helper_funtions import get_project_root
from typing import List, Optional
import os


class ModelConfig(BaseModel):
    num_phon_enc_layers: PositiveInt = Field(
        default=8, description="Number of transformer layers in the phonology encoder."
    )
    num_orth_enc_layers: PositiveInt = Field(
        default=1, description="Number of transformer layers in the orthography encoder."
    )
    num_mixing_enc_layers: PositiveInt = Field(
        default=2, description="Number of transformer layers in the mixing encoder."
    )
    num_phon_dec_layers: PositiveInt = Field(
        default=4, description="Number of transformer layers in the phonology decoder."
    )
    num_orth_dec_layers: PositiveInt = Field(
        default=1, description="Number of transformer layers in the orthography decoder."
    )
    d_model: PositiveInt = Field(default=128, description="Dimensionality of the model.")
    nhead: PositiveInt = Field(default=16, description="Number of attention heads.")
    d_embedding: PositiveInt = Field(default=1, description="Dimensionality of the embedding.")
    seed: Optional[PositiveInt] = Field(default=None, description="Random seed for reproducibility.")

    @field_validator("d_model")
    def validate_d_model(cls, v, info: ValidationInfo):
        nhead = info.data.get("nhead")
        if nhead is not None and v % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        return v

    class Config:
        protected_namespaces = ()
