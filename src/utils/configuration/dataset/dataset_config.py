from pydantic import BaseModel, Field, field_validator, ValidationInfo, model_validator
from src.utils.helper_funtions import get_project_root
from typing import List, Optional
import os


class DatasetConfig(BaseModel):
    dataset_filepath: str = Field(description="Name of the dataset file")

    @model_validator(mode="before")
    def convert_paths(cls, values):
        """Convert relative paths to absolute paths before validation occurs."""
        project_root = get_project_root()
        if "dataset_filepath" not in values:
            values["dataset_filepath"] = cls.model_fields["dataset_filepath"].get_default()

        values["dataset_filepath"] = os.path.join(project_root, "data", values["dataset_filepath"])
        return values

    @model_validator(mode="after")
    def validate_paths(self):
        if not os.path.exists(self.dataset_filepath):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_filepath}")

        return self
