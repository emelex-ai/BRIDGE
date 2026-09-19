import os

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetConfig(BaseModel):
    # An unknown key is a typo in someone's experiment config, and silently
    # ignoring it means the run does something other than what the file says.
    # Removing a field (gcs_path, max_nb_steps) made that reachable.
    model_config = ConfigDict(extra="forbid")

    dataset_filepath: str = Field(description="Path to dataset file")
    custom_cmudict_path: str | None = Field(
        default=None,
        description=(
            "Optional path to a custom CMU dictionary file. The dictionary should follow the "
            "nested-by-language shape `{word: {lang_code: [[phonemes]]}}` and is merged on top "
            "of the lexicons shipped under `bridge/core/pronunciation_lexicons/`."
        ),
    )

    @model_validator(mode="after")
    def validate_paths(self):
        if "gs://" not in self.dataset_filepath:
            if not os.path.exists(self.dataset_filepath):
                raise FileNotFoundError(f"Dataset file not found: {self.dataset_filepath}")
        return self
