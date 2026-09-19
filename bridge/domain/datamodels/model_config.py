from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from bridge.domain.datamodels.vocab_spec import VocabSpec


class ModelConfig(BaseModel):
    # An unknown key is a typo in someone's experiment config, and silently
    # ignoring it means the run does something other than what the file says.
    # Removing a field (gcs_path, max_nb_steps) made that reachable.
    model_config = ConfigDict(extra="forbid")

    num_phon_enc_layers: int = Field(default=2)
    num_orth_enc_layers: int = Field(default=2)
    num_mixing_enc_layers: int = Field(default=2)
    num_phon_dec_layers: int = Field(default=2)
    num_orth_dec_layers: int = Field(default=2)
    d_model: int = Field(default=64)
    nhead: int = Field(default=2)
    d_embedding: int = Field(default=1)
    seed: int | None = Field(default=None)
    # Build via `VocabSpec.from_tokenizer(bridge_tokenizer)`.
    vocab: VocabSpec

    @field_validator("d_model")
    def validate_d_model(cls, v, info: ValidationInfo):
        nhead = info.data.get("nhead")
        if nhead is not None and v % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        return v
