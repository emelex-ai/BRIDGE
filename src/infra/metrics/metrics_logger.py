import os
from typing import Literal
from src.domain.datamodels.metrics_config import MetricsConfig, OutputMode
from abc import ABC, abstractmethod
import torch


class MetricsLogger(ABC):
    def __init__(self, metrics_config: MetricsConfig) -> None:
        self.metrics_config = metrics_config

    @abstractmethod
    def log_metrics(self, metrics: dict, level: Literal["BATCH", "EPOCH"]) -> None:
        raise NotImplementedError


class STDOutMetricsLogger(MetricsLogger):
    def log_metrics(self, metrics: dict, level: Literal["BATCH", "EPOCH"]) -> None:
        print(level, metrics)


class MultipleMetricsLogger(MetricsLogger):
    def __init__(
        self, metrics_config: MetricsConfig, loggers: list[MetricsLogger]
    ) -> None:
        super().__init__(metrics_config)
        self.loggers = loggers
    
    def log_metrics(self, metrics: dict, level: Literal["BATCH", "EPOCH"]) -> None:
        for logger in self.loggers:
            logger.log_metrics(metrics, level)


class CSVMetricsLogger(MetricsLogger):

    def __init__(self, metrics_config: MetricsConfig):
        # Create results directory if it doesn't exist
        if not os.path.exists("results"):
            os.makedirs("results")
        super().__init__(metrics_config)
        self.opened = {
                       "BATCH": False,
            "EPOCH": False
        }

    
    def log_metrics(self, metrics: dict, level: Literal["BATCH", "EPOCH"]) -> None:
        if not self.metrics_config.filename:
            raise ValueError("Filename is required for CSV output mode")
        for metric in metrics:
            if isinstance(metrics[metric], torch.Tensor):
                metrics[metric] = metrics[metric].item()
        # First log should include header columns
        if not self.opened[level]:
            with open(f"results/{self.metrics_config.filename.split(".")[0]}_{level}.{self.metrics_config.filename.split(".")[1]}", "w") as f:
                f.write(",".join(metrics.keys()) + "\n")
                f.write(",".join([str(v) for v in metrics.values()]) + "\n")
            self.opened[level] = True
        else:
            with open(f"results/{self.metrics_config.filename.split(".")[0]}_{level}.{self.metrics_config.filename.split(".")[1]}", "a") as f:
                f.write(",".join([str(v) for v in metrics.values()]) + "\n")


def metrics_logger_factory(metrics_config: MetricsConfig) -> MetricsLogger:
    loggers = []
    if OutputMode.CSV in metrics_config.modes:
        if metrics_config.filename:
            loggers.append(CSVMetricsLogger(metrics_config))
        else:
            raise ValueError("Filename is required for CSV output mode")
    if OutputMode.STDOUT in metrics_config.modes:
        loggers.append(STDOutMetricsLogger(metrics_config))
    return MultipleMetricsLogger(metrics_config, loggers)
