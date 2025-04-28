from pydantic import BaseModel, Field, field_validator, ValidationInfo
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
    seed: int | None = Field(default=None)

    @field_validator("d_model")
    def validate_d_model(cls, v, info: ValidationInfo):
        nhead = info.data.get("nhead")
        if nhead is not None and v % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        return v
