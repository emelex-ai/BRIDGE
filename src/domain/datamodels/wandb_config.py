from pydantic import BaseModel, Field, field_validator, ValidationInfo, model_validator
from src.utils.helper_functions import get_project_root
from typing import List, Optional
import os
from pydantic import ConfigDict

class WandbConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    project: str = Field(description="Name of the project")
    entity: str = Field(description="Name of the entity")
    is_enabled: bool = Field(
        default=False, description="Flag to enable or disable wandb"
    )
