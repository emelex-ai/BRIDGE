from src.domain.datamodels.metrics_config import MetricsConfig, OutputMode
from abc import ABC, abstractmethod
import torch


class MetricsLogger(ABC):
    def __init__(self, metrics_config: MetricsConfig) -> None:
        self.metrics_config = metrics_config

    @abstractmethod
    def log_metrics(self, metrics: dict) -> None:
        raise NotImplementedError
    



class STDOutMetricsLogger(MetricsLogger):
    def log_metrics(self, metrics: dict) -> None:
        print(metrics)


class MultipleMetricsLogger(MetricsLogger):
    def __init__(self, metrics_config: MetricsConfig, loggers: list[MetricsLogger]) -> None:
        super().__init__(metrics_config)
        self.loggers = loggers
    
    def log_metrics(self, metrics: dict) -> None:
        for logger in self.loggers:
            logger.log_metrics(metrics)



class CSVMetricsLogger(MetricsLogger):
    
    def __init__(self, metrics_config: MetricsConfig):
        super().__init__(metrics_config)
        self.opened = False

    
    def log_metrics(self, metrics: dict) -> None:
        if not self.metrics_config.filename:
            raise ValueError("Filename is required for CSV output mode")
        for metric in metrics:
            if isinstance(metrics[metric], torch.Tensor):
                metrics[metric] = metrics[metric].item()
        # First log should include header columns
        if not self.opened:
            with open(self.metrics_config.filename, "w") as f:
                f.write(",".join(metrics.keys()) + "\n")
                f.write(",".join([str(v) for v in metrics.values()]) + "\n")
            self.opened = True
        else:
            with open(self.metrics_config.filename, "a") as f:
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
    else:
        raise ValueError(f"Unsupported output mode: {metrics_config.modes}")
    return MultipleMetricsLogger(metrics_config, loggers)
        