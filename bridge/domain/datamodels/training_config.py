import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bridge.domain.datamodels.pathways import Pathway


class TrainingConfig(BaseModel):
    # An unknown key is a typo in someone's experiment config, and silently
    # ignoring it means the run does something other than what the file says.
    # Removing a field (gcs_path, max_nb_steps) made that reachable.
    model_config = ConfigDict(extra="forbid")

    learning_rate: float = Field(default=0.001)
    training_pathway: Pathway = Field(default="o2p")
    model_artifacts_dir: str = Field(default="model_artifacts")
    weight_decay: float = Field(default=0.0)
    checkpoint_path: str | None = Field(default=None)

    @model_validator(mode="before")
    def convert_paths(cls, values):
        """Resolve relative paths, without touching the filesystem.

        A relative ``model_artifacts_dir`` resolves against the caller's working directory.
        It used to be joined onto ``get_project_root()``, which is BRIDGE's own install
        root, so ``model_artifacts_dir="runs/experiment1"`` wrote checkpoints inside the
        installed package where nobody looks and the next reinstall deletes them. Absolute
        paths happened to work only because ``os.path.join`` discards its prefix when the
        second argument is absolute, so behaviour turned on a property of the string that
        was never documented.

        Validation no longer creates the directory either. Constructing a config is not a
        reason to write to disk, and merely validating one in a test created directories.
        ``TrainingPipeline.save_checkpoint`` creates it at the point of first write instead.

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
        """A missing checkpoint is an error; a missing artifacts directory is not.

        The artifacts directory is created lazily on first write, so its absence at
        construction says nothing. A checkpoint that does not exist is a genuine mistake
        and is worth catching before a run starts rather than after it has trained.
        """
        if self.checkpoint_path and not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.checkpoint_path}")
        return self
