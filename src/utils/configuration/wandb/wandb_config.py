from pydantic import BaseModel, Field, field_validator, ValidationInfo, model_validator
from src.utils.helper_funtions import get_project_root
from typing import List, Optional
import os


class WandbConfig(BaseModel):
    project: str = Field(description="Name of the project")
    wandb_enabled: bool = Field(default=False, description="Flag to enable or disable wandb")
    sweep_filename: Optional[str] = Field(default=None)

    @model_validator(mode="before")
    def convert_paths(cls, values):
        """Convert relative paths to absolute paths before validation occurs."""
        project_root = get_project_root()
        values.setdefault("sweep_filename", cls.model_fields["sweep_filename"].get_default())

        if values.get("sweep_filename"):
            values["sweep_filename"] = os.path.join(project_root, values["sweep_filename"])

        return values

    @model_validator(mode="after")
    def validate_paths(self):
        if self.sweep_filename and not os.path.exists(self.sweep_filename):
            raise FileNotFoundError(f"Sweep file not found: {self.sweep_filename}")
