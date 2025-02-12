from pydantic import BaseModel, Field, model_validator, PositiveInt, ConfigDict
from src.utils.helper_funtions import get_project_root
import os


class DatasetConfig(BaseModel):
    dataset_filepath: str = Field(description="")
    dimension_phon_repr: PositiveInt = Field(
        description="Length of vector of the phonological representation"
    )
    orthographic_vocabulary_size: PositiveInt | None = Field(
        default=None, gt=0, description="Orthographic Vocabulary Size"
    )
    phonological_vocabulary_size: PositiveInt | None = Field(
        default=None, gt=0, description="Phonological Vocabulary Size"
    )
    max_orth_seq_len: PositiveInt | None = Field(
        default=None, gt=0, description="Maximum length of the orthographic sequence"
    )
    max_phon_seq_len: PositiveInt | None = Field(
        default=None, gt=0, description="Maximum length of the phonological sequence"
    )

    @model_validator(mode="before")
    def convert_paths(cls, values):
        """Convert relative paths to absolute paths before validation occurs."""
        project_root = get_project_root()
        if "dataset_filepath" not in values:
            values["dataset_filepath"] = cls.model_fields[
                "dataset_filepath"
            ].get_default()

        values["dataset_filepath"] = os.path.join(
            project_root, "data", values["dataset_filepath"]
        )
        return values

    @model_validator(mode="after")
    def validate_paths(self):
        if not os.path.exists(self.dataset_filepath):
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_filepath}")
        return self
