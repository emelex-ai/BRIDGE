from pydantic import BaseModel, Field, model_validator, PositiveInt
from src.utils.helper_funtions import get_project_root
from typing import Optional
import os


class DatasetConfig(BaseModel):
    dataset_filepath: str = Field(description="Path to the primary dataset file.")
    dimension_phon_repr: PositiveInt = Field(description="Dimensionality of the phonological representation.")
    orthographic_vocabulary_size: Optional[PositiveInt] = Field(
        default=None, description="Size of the orthographic vocabulary."
    )
    phonological_vocabulary_size: Optional[PositiveInt] = Field(
        default=None, description="Size of the phonological vocabulary."
    )
    max_orth_seq_len: Optional[PositiveInt] = Field(
        default=None, description="Maximum sequence length for orthography."
    )
    max_phon_seq_len: Optional[PositiveInt] = Field(default=None, description="Maximum sequence length for phonology.")

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
