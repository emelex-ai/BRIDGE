from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bridge.domain.datamodels import (
    DatasetConfig,
    MetricsConfig,
    ModelConfig,
    TrainingConfig,
    WandbConfig,
)


def load_all_configs() -> dict[str, Any]:
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


class BridgeConfig(BaseModel):
    """Combined configuration class for the BRIDGE model.

    This class combines all individual configuration classes into a single
    comprehensive configuration object for easier management and access.
    """

    model_config = ConfigDict(validate_assignment=True)

    # Individual config sections
    wandb_config: WandbConfig = Field(
        description="WandB experiment tracking configuration"
    )
    model_settings: ModelConfig = Field(description="Model architecture configuration")
    dataset_config: DatasetConfig = Field(
        description="Dataset and data processing configuration"
    )
    training_config: TrainingConfig = Field(
        description="Training hyperparameters and settings"
    )
    metrics_config: MetricsConfig = Field(
        description="Metrics logging and evaluation configuration"
    )

    @model_validator(mode="after")
    def validate_cross_config_consistency(self):
        """Validate consistency across different configuration sections."""
        # Ensure sequence lengths are consistent between model and dataset configs
        if self.model_settings.max_orth_seq_len != self.model_settings.max_orth_seq_len:
            raise ValueError(
                f"Orthographic sequence length mismatch: "
                f"model_settings.max_orth_seq_len={self.model_settings.max_orth_seq_len} "
                f"vs dataset_config.max_orth_seq_len={self.dataset_config.max_orth_seq_len}"
            )

        if self.model_settings.max_phon_seq_len != self.model_settings.max_phon_seq_len:
            raise ValueError(
                f"Phonological sequence length mismatch: "
                f"model_settings.max_phon_seq_len={self.model_settings.max_phon_seq_len} "
                f"vs dataset_config.max_phon_seq_len={self.dataset_config.max_phon_seq_len}"
            )

        return self

    def get_wandb_config(self) -> dict[str, Any]:
        """Get WandB configuration as a dictionary for experiment tracking.

        Returns:
            Dictionary containing model and dataset configurations for WandB.
        """
        return {
            **self.model_settings.model_dump(),
            **self.dataset_config.model_dump(),
            **self.training_config.model_dump(),
        }

    def update_from_dict(self, updates: dict[str, Any]) -> None:
        """Update configuration from a dictionary.

        Args:
            updates: Dictionary containing configuration updates.
        """
        for key, value in updates.items():
            if hasattr(self.model_settings, key):
                setattr(self.model_settings, key, value)
            elif hasattr(self.training_config, key):
                setattr(self.training_config, key, value)
            elif hasattr(self.dataset_config, key):
                setattr(self.dataset_config, key, value)
            elif hasattr(self.wandb_config, key):
                setattr(self.wandb_config, key, value)
            elif hasattr(self.metrics_config, key):
                setattr(self.metrics_config, key, value)


def load_configs() -> BridgeConfig:
    """Load all configurations and return a combined BridgeConfig object.

    This function loads all individual configuration files and combines them
    into a single BridgeConfig object with cross-validation.

    Returns:
        BridgeConfig: A combined configuration object containing all settings.
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

    # Create combined configuration object
    return BridgeConfig(
        wandb_config=configs["wandb_config"],
        model_settings=configs["model_config"],
        dataset_config=configs["dataset_config"],
        training_config=configs["training_config"],
        metrics_config=configs["metrics_config"],
    )


def load_configs_dict() -> dict[str, Any]:
    """Load the configurations for the application (legacy function).

    This function is maintained for backward compatibility.

    Returns:
        dict: A dictionary containing the configurations for the application.
    """
    bridge_config = load_configs()
    return {
        "wandb_config": bridge_config.wandb_config,
        "model_config": bridge_config.model_settings,
        "dataset_config": bridge_config.dataset_config,
        "training_config": bridge_config.training_config,
        "metrics_config": bridge_config.metrics_config,
    }


if __name__ == "__main__":
    # Test the configuration loading
    print("Testing BridgeConfig loading...")

    try:
        config = load_configs()
        print("✓ Configuration loaded successfully")
        print(
            f"✓ Model config: d_model={config.model_settings.d_model}, nhead={config.model_settings.nhead}"
        )
        print(
            f"✓ Training config: epochs={config.training_config.num_epochs}, lr={config.training_config.learning_rate}"
        )
        print(f"✓ Dataset config: filepath={config.dataset_config.dataset_filepath}")
        print(
            f"✓ WandB config: project={config.wandb_config.project}, enabled={config.wandb_config.is_enabled}"
        )
        print(f"✓ Metrics config: modes={config.metrics_config.modes}")

        # Test cross-config validation
        print("✓ Cross-config validation passed")

        # Test WandB config generation
        wandb_dict = config.get_wandb_config()
        print(f"✓ WandB config dict generated with {len(wandb_dict)} keys")

        print("All tests passed!")

    except Exception as e:
        print(f"✗ Configuration loading failed: {e}")
        raise
