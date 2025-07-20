from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class ModelConfig(BaseModel):
    # Enable validation on assignment
    model_config = ConfigDict(validate_assignment=True)
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

    is_causal: bool = Field(
        default=False, description="Whether to apply causal masking (for encoders)"
    )

    max_seq_len: int = Field(
        default=4096, description="Maximum sequence length for pre-computed masks"
    )

    max_orth_seq_len: int = Field(
        default=983,
        description="Maximum sequence length for orthography pre-computed masks",
    )

    max_phon_seq_len: int = Field(
        default=883,
        description="Maximum sequence length for phonology pre-computed masks",
    )

    ensure_contiguous: bool = Field(
        default=False,
        description="Whether to ensure sliced masks are contiguous for GPU performance",
    )

    @field_validator("max_seq_len")
    def validate_max_seq_len(cls, v: int) -> int | None:
        if v <= 0:
            raise ValueError("max_seq_len must be positive")
        if v > 65536:  # Reasonable upper limit
            raise ValueError("max_seq_len should not exceed 65536")
        return v

    @field_validator("max_orth_seq_len")
    def validate_max_orth_seq_len(cls, v: int) -> int | None:
        if v <= 0:
            raise ValueError("max_orth_seq_len must be positive")
        if v > 4096:  # Reasonable upper limit
            raise ValueError("max_orth_seq_len should not exceed 4096")
        return v

    @field_validator("max_phon_seq_len")
    def validate_max_phon_seq_len(cls, v: int) -> int | None:
        if v <= 0:
            raise ValueError("max_phon_seq_len must be positive")
        if v > 4096:  # Reasonable upper limit
            raise ValueError("max_phon_seq_len should not exceed 4096")
        return v

    @field_validator("window_size")
    def validate_window_size(cls, v: int) -> int | None:
        if v <= 0:
            raise ValueError("window_size must be positive")
        if v % 2 == 0:
            raise ValueError("window_size should be odd for symmetric windows")
        return v
