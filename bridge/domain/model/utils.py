from typing import Any


def load_configs() -> dict[str, Any]:
    """Load the configurations for the application.

    This function is used to load the configurations for the application.

    Returns:
        dict: A dictionary containing the configurations for the application.
    """
    from bridge.application.handlers import (
        DatasetConfigHandler,
        MetricsConfigHandler,
        ModelConfigHandler,
        TrainingConfigHandler,
        WandbConfigHandler,
    )

    # Centralized config loading
    handlers = {
        "wandb_config": WandbConfigHandler,
        "model_config": ModelConfigHandler,
        "dataset_config": DatasetConfigHandler,
        "training_config": TrainingConfigHandler,
        "metrics_config": MetricsConfigHandler,
    }
    configs = {}
    for key, handler_cls in handlers.items():
        handler = handler_cls(config_filepath=f"app/config/{key}.yaml")
        # handler.print_config()
        configs[key] = handler.get_config()
    return configs
