from pydantic import BaseModel, ConfigDict, Field


class WandbConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    project: str = Field(description="Name of the project")
    entity: str = Field(description="Name of the entity")
    is_enabled: bool = Field(default=False, description="Flag to enable or disable wandb")
