import os

from pydantic import BaseModel, Field, model_validator

from bridge.utils.helper_functions import get_project_root


class DatasetConfig(BaseModel):
    dataset_filepath: str = Field(description="Path to dataset file")
    custom_cmudict_path: str = Field(
        default=os.path.join(get_project_root(), "bridge/core/custom_cmudict.json"),
        description="Path to custom CMU dictionary file",
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
