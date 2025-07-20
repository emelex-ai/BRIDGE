
def load_configs():  # W: Missing function or method docstring
    # Move imports here to avoid a circular import dependency.
    from bridge.application.handlers import (
        DatasetConfigHandler,
        LoggingConfigHandler,
        MetricsConfigHandler,
        ModelConfigHandler,
        TrainingConfigHandler,
        TrainModelHandler,
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
