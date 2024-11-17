from pydantic import BaseModel, Field, model_validator, PositiveInt
from src.utils.helper_funtions import get_project_root
from typing import Optional
import os


class DatasetConfig(BaseModel):
    dataset_filepath: str = Field(description="")
    dimension_phon_repr: PositiveInt = Field(default=0, description="Length of vector of the phonological representation")
    orthographic_vocabulary_size: Optional[int] = Field(default=None, description="Orthographic Vocabulary Size")
    phonological_vocabulary_size: Optional[int] = Field(default=None, description="Phonological Vocabulary Size")
    max_orth_seq_len: Optional[int] = Field(default=0, description="")
    max_phon_seq_len: Optional[int] = Field(default=0, description="")

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

    def validate_vocab_sizes(self):
        """Ensure vocabulary sizes are set before they are accessed."""
        if self.orthographic_vocabulary_size is None:
            raise ValueError("Orthographic vocabulary size must be set before using this configuration.")
        if self.phonological_vocabulary_size is None:
            raise ValueError("Phonological vocabulary size must be set before using this configuration.")
