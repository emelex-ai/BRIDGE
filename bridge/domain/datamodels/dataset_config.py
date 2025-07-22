import os
from pathlib import PosixPath

from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator

from bridge.utils import get_project_root


class DatasetConfig(BaseModel):
    # Enable validation on assignment
    model_config = ConfigDict(validate_assignment=True)

    dataset_filepath: str | PosixPath = Field(description="Path to dataset file")
    num_samples: int = Field(default=100, description="Number of training+validation+testing samples")
    custom_cmudict_path: str | PosixPath = Field(
        default=None, description="Path to custom CMU dictionary file"
    )
    tokenizer_cache_size: int = Field(
        default=10000, description="Max cache size for tokenizer"
    )

    @model_validator(mode="before")
    def convert_paths(cls, values):
        """Convert relative paths to absolute paths before validation occurs."""
        project_root = get_project_root()
        if "dataset_filepath" not in values:
            values["dataset_filepath"] = cls.model_fields[
                "dataset_filepath"
            ].get_default()

        if "gs://" not in values["dataset_filepath"]:
            # Convert relative paths to absolute paths
            values["dataset_filepath"] = os.path.join(
                project_root, "data", values["dataset_filepath"]
            )

        if "custom_cmudict_path" in values and values["custom_cmudict_path"]:
            values["custom_cmudict_path"] = os.path.join(
                project_root, "data", values["custom_cmudict_path"]
            )

        # For backward compatibility
        if "phoneme_cache_size" in values and "tokenizer_cache_size" not in values:
            values["tokenizer_cache_size"] = values["phoneme_cache_size"]

        return values

    @model_validator(mode="after")
    def validate_paths(self):
        if "gs://" not in self.dataset_filepath:
            if not os.path.exists(self.dataset_filepath):
                raise FileNotFoundError(
                    f"Dataset file not found: {self.dataset_filepath}"
                )
        return self

    @field_validator("num_samples")
    def validate_num_samples(cls, v: int) -> int | None:
        # if v == -1:
        #   Read the entire data file. But not implemented for the synthetic file. 
        if v < -1:
            raise ValueError("window_size must be positive")
        return v
