import json
import os
from typing import Literal
from src.domain.datamodels.metrics_config import MetricsConfig, OutputMode
from abc import ABC, abstractmethod
from src.infra.clients.gcp.gcs_client import GCSClient
import torch


class MetricsLogger(ABC):
    def __init__(self, metrics_config: MetricsConfig) -> None:
        self.metrics_config = metrics_config

    @abstractmethod
    def log_metrics(self, metrics: dict, level: Literal["BATCH", "EPOCH"]) -> None:
        """
        Log metrics to the specified output.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self) -> None:
        """
        Some loggers may need to save their state or close files after logging.
        """
        raise NotImplementedError


class STDOutMetricsLogger(MetricsLogger):
    def log_metrics(self, metrics: dict, level: Literal["BATCH", "EPOCH"]) -> None:
        print(level, metrics)

    def save(self) -> None:
        pass


class MultipleMetricsLogger(MetricsLogger):
    def __init__(
        self, metrics_config: MetricsConfig, loggers: list[MetricsLogger]
    ) -> None:
        super().__init__(metrics_config)
        self.loggers = loggers

    def log_metrics(self, metrics: dict, level: Literal["BATCH", "EPOCH"]) -> None:
        for logger in self.loggers:
            logger.log_metrics(metrics, level)

    def save(self) -> None:
        for logger in self.loggers:
            logger.save()
        print("Finished logging metrics to all configured outputs.")


class CSVMetricsLogger(MetricsLogger):

    def __init__(self, metrics_config: MetricsConfig):
        # Create results directory if it doesn't exist
        if not os.path.exists("results"):
            os.makedirs("results")
        super().__init__(metrics_config)
        self.opened = {"BATCH": False, "EPOCH": False}
        for level in ["BATCH", "EPOCH"]:
            if self.metrics_config.filename:
                file_name = f"results/{self.metrics_config.filename.split('.')[0]}_{level}.{self.metrics_config.filename.split('.')[1]}"
                if os.path.exists(file_name):
                    self.opened[level] = True
                    print(f"Found existing metrics file: {file_name}")

    def log_metrics(self, metrics: dict, level: Literal["BATCH", "EPOCH"]) -> None:
        if not self.metrics_config.filename:
            raise ValueError("Filename is required for CSV output mode")
        for metric in metrics:
            if isinstance(metrics[metric], torch.Tensor):
                metrics[metric] = metrics[metric].item()
        # First log should include header columns
        file_name = f"results/{self.metrics_config.filename.split('.')[0]}_{level}.{self.metrics_config.filename.split('.')[1]}"
        if not self.opened[level]:
            with open(file_name, "w") as f:
                f.write(",".join(metrics.keys()) + "\n")
                f.write(",".join([json.dumps(v) for v in metrics.values()]) + "\n")
            self.opened[level] = True
        else:
            with open(file_name, "a") as f:
                f.write(",".join([json.dumps(v) for v in metrics.values()]) + "\n")

    def save(self) -> None:
        pass


class CSVGCPMetricsLogger(CSVMetricsLogger):
    def __init__(self, metrics_config: MetricsConfig, gcs_client: GCSClient) -> None:
        super().__init__(metrics_config)
        self.gcs_client = gcs_client

    def save(self) -> None:
        index = int(os.environ["CLOUD_RUN_TASK_INDEX"]) + 1
        pretraining_index = index // 22 + 1
        finetuning_index = index % 22
        for level in self.opened:
            if self.opened[level]:
                file_name = f"results/{self.metrics_config.filename.split('.')[0]}_{level}.{self.metrics_config.filename.split('.')[1]}"
                self.gcs_client.upload_file(os.environ["BUCKET_NAME"], file_name, f"finetuning/{finetuning_index}/{pretraining_index}/results/{file_name}")


def metrics_logger_factory(metrics_config: MetricsConfig) -> MetricsLogger:
    loggers = []
    if OutputMode.CSV in metrics_config.modes:
        if metrics_config.filename:
            loggers.append(CSVMetricsLogger(metrics_config))
        else:
            raise ValueError("Filename is required for CSV output mode")
    if OutputMode.STDOUT in metrics_config.modes:
        loggers.append(STDOutMetricsLogger(metrics_config))
    if OutputMode.GCS in metrics_config.modes:
        loggers.append(CSVGCPMetricsLogger(metrics_config, GCSClient()))
    return MultipleMetricsLogger(metrics_config, loggers)
