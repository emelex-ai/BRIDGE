import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bridge.domain.datamodels.pathways import Pathway


class TrainingConfig(BaseModel):
    # An unknown key is a typo in an experiment config. Ignored, the run does something
    # other than what the file says.
    model_config = ConfigDict(extra="forbid")

    learning_rate: float = Field(default=0.001)
    training_pathway: Pathway = Field(default="o2p")
    model_artifacts_dir: str = Field(default="model_artifacts")
    weight_decay: float = Field(default=0.0)
    checkpoint_path: str | None = Field(default=None)

    @model_validator(mode="before")
    def convert_paths(cls, values):
        """Resolve a relative ``model_artifacts_dir`` against the caller's working directory.

        Not against ``get_project_root()``: that is BRIDGE's install root, where a
        reinstall deletes whatever a run wrote. The directory is not created here, since
        building a config writes nothing to disk. ``TrainingPipeline.save_checkpoint``
        creates it on first write.
        """
        values.setdefault(
            "model_artifacts_dir", cls.model_fields["model_artifacts_dir"].get_default()
        )
        artifacts = Path(values["model_artifacts_dir"])
        values["model_artifacts_dir"] = str(
            artifacts if artifacts.is_absolute() else Path.cwd() / artifacts
        )
        return values

    @model_validator(mode="after")
    def validate_paths(self):
        """A missing checkpoint raises. A missing artifacts directory does not: it is
        created on first write."""
        if self.checkpoint_path and not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
        return self
