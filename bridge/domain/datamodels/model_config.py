from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


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

    @field_validator("d_model")
    def validate_d_model(cls, v, info: ValidationInfo):
        nhead = info.data.get("nhead")
        if nhead is not None and v % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        return v

    # Sliding window configuration
    window_size: int = Field(
        default=61, description="±30 characters + current position"
    )

    use_sliding_window: bool = Field(
        default=False, description="Enable sliding window attention"
    )
