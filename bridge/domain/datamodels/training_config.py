import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bridge.domain.datamodels.pathways import Pathway
from bridge.utils import get_project_root


class TrainingConfig(BaseModel):
    # An unknown key is a typo in someone's experiment config, and silently
    # ignoring it means the run does something other than what the file says.
    # Removing a field (gcs_path, max_nb_steps) made that reachable.
    model_config = ConfigDict(extra="forbid")

    num_epochs: int = Field(default=2)
    batch_size_train: int = Field(default=32)
    batch_size_val: int = Field(default=32)
    train_test_split: float = Field(default=0.8, ge=0.0, le=1.0)
    learning_rate: float = Field(default=0.001)
    training_pathway: Pathway = Field(default="o2p")
    model_artifacts_dir: str = Field(default="model_artifacts")
    weight_decay: float = Field(default=0.0)
    checkpoint_path: str | None = Field(default=None)
    test_data_path: str | None = Field(default=None)
    num_chunks: int | None = Field(
        default=1,
        description="Number of chunks to split a batch into for accumulated gradients",
    )
    seed: int | None = Field(
        default=None,
        description="Seeds the per-epoch training-order shuffle. None leaves it unseeded.",
    )
    compute_metrics: bool = Field(
        default=False,
        description=(
            "Score accuracy and distance metrics alongside the loss, on every step. Off by "
            "default: the phonological metrics cost ~7 ms per step and the loss is always "
            "reported. See docs/decisions/0005 for what the phonological ones mean."
        ),
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
