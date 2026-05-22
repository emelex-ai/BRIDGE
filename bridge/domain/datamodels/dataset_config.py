import os

from pydantic import BaseModel, Field, model_validator


class DatasetConfig(BaseModel):
    dataset_filepath: str = Field(description="Path to dataset file")
    custom_cmudict_path: str | None = Field(
        default=None,
        description=(
            "Optional path to a custom CMU dictionary file. The dictionary should follow the "
            "nested-by-language shape `{word: {lang_code: [[phonemes]]}}` and is merged on top "
            "of the lexicons shipped under `bridge/core/pronunciation_lexicons/`."
        ),
    )
    tokenizer_cache_size: int = Field(default=10000, description="Max cache size for tokenizer")

    @model_validator(mode="before")
    def convert_paths(cls, values):
        """Convert relative paths to absolute paths before validation occurs."""
        if "dataset_filepath" not in values:
            raise FileNotFoundError("No dataset file specified")

        # For backward compatibility
        if "phoneme_cache_size" in values and "tokenizer_cache_size" not in values:
            values["tokenizer_cache_size"] = values["phoneme_cache_size"]

        return values

    @model_validator(mode="after")
    def validate_paths(self):
        if "gs://" not in self.dataset_filepath:
            if not os.path.exists(self.dataset_filepath):
                raise FileNotFoundError(f"Dataset file not found: {self.dataset_filepath}")
        return self
