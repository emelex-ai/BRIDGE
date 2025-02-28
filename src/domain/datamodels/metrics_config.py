from enum import StrEnum
from pydantic import BaseModel

class OutputMode(StrEnum):
    CSV = "csv"
    STDOUT = "stdout"

class MetricsConfig(BaseModel):
    batch_metrics: bool
    training_metrics: bool
    validation_metrics: bool
    mode: OutputMode
    filename: str | None