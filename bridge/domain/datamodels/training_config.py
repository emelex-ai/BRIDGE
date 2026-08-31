import os
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

from bridge.utils import get_project_root


class TrainingConfig(BaseModel):
    num_epochs: int = Field(default=2)
    batch_size_train: int = Field(default=32)
    batch_size_val: int = Field(default=32)
    train_test_split: float = Field(default=0.8)
    max_nb_steps: int | None = Field(default=None)
    learning_rate: float = Field(default=0.001)
    training_pathway: str = Field(default="o2p")
    model_artifacts_dir: str = Field(default="model_artifacts")
    weight_decay: float = Field(default=0.0)
    checkpoint_path: str | None = Field(default=None)
    test_data_path: str | None = Field(default=None)
    num_chunks: int | None = Field(
        default=1,
        description="Number of chunks to split a batch into for accumulated gradients",
    )
    gcs_path: str | None = Field(default=None)
    seed: int | None = Field(
        default=None,
        description="Seeds the per-epoch training-order shuffle. None leaves it unseeded.",
    )
    shuffle_each_epoch: bool = Field(
        default=True,
        description=(
            "Reorder the training partition between epochs. Defaults to True: defaulting to "
            "False would preserve the defect this was added to fix."
        ),
    )

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

        ``test_data_path`` still resolves under ``<project root>/data``, unchanged.
        """
        project_root = get_project_root()
        values.setdefault(
            "model_artifacts_dir", cls.model_fields["model_artifacts_dir"].get_default()
        )
        artifacts = Path(values["model_artifacts_dir"])
        values["model_artifacts_dir"] = str(
            artifacts if artifacts.is_absolute() else Path.cwd() / artifacts
        )
        if "test_data_path" in values and values["test_data_path"]:
            values["test_data_path"] = os.path.join(project_root, "data", values["test_data_path"])
        return values

    @field_validator("training_pathway")
    def validate_pathway(cls, v: str) -> str:
        allowed_training_pathways = ["o2p", "p2o", "op2op", "p2p"]
        if v not in allowed_training_pathways:
            raise ValueError(f"Invalid pathway: {v}. Allowed: {allowed_training_pathways}")
        return v

    @field_validator("train_test_split")
    def validate_train_test_split(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("train_test_split must be between 0.0 and 1.0")
        return v

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
