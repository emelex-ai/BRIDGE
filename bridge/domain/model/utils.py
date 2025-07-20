import pprint
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bridge.domain.datamodels import (
    DatasetConfig,
    MetricsConfig,
    ModelConfig,
    TrainingConfig,
    WandbConfig,
)


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
        # Only validate if both configs have explicit sequence length settings
        # This allows for flexible configuration where one config can use defaults
        model_has_orth_len = hasattr(self.model_settings, "max_orth_seq_len")
        dataset_has_orth_len = hasattr(self.dataset_config, "max_orth_seq_len")
        model_has_phon_len = hasattr(self.model_settings, "max_phon_seq_len")
        dataset_has_phon_len = hasattr(self.dataset_config, "max_phon_seq_len")

        # Only validate if both configs explicitly set the same parameter
        if (
            model_has_orth_len
            and dataset_has_orth_len
            and self.model_settings.max_orth_seq_len
            != self.dataset_config.max_orth_seq_len
        ):
            raise ValueError(
                f"Orthographic sequence length mismatch: "
                f"model_settings.max_orth_seq_len={self.model_settings.max_orth_seq_len} "
                f"vs dataset_config.max_orth_seq_len={self.dataset_config.max_orth_seq_len}"
            )

        if (
            model_has_phon_len
            and dataset_has_phon_len
            and self.model_settings.max_phon_seq_len
            != self.dataset_config.max_phon_seq_len
        ):
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

    def print_config(self, indent: int = 0) -> None:
        """Print the configuration in a pretty format.

        Args:
            indent: Number of spaces to indent each level (default: 0).
        """
        print_config_pretty(self, indent)

    def print_scalable_config(self, indent: int = 0) -> None:
        """Print the configuration in a scalable, maintainable format.

        Args:
            indent: Number of spaces to indent each level (default: 0).
        """
        print_scalable_config_pretty(self, indent)


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


def print_scalable_config_pretty(config: BridgeConfig, indent: int = 0) -> None:
    """Print all configuration parameters in a scalable, maintainable format.

    This function automatically adapts to any configuration changes by iterating
    over the configuration objects and converting them to dictionaries for display.

    Args:
        config: The BridgeConfig object to print.
        indent: Number of spaces to indent each level (default: 0).
    """
    indent_str = " " * indent

    print(
        f"{indent_str}╔══════════════════════════════════════════════════════════════"
    )
    print(f"{indent_str}║ BRIDGE MODEL CONFIGURATION")
    print(
        f"{indent_str}╚══════════════════════════════════════════════════════════════"
    )

    # Configuration sections with their display names and emojis
    config_sections = [
        ("wandb_config", "📊 WandB Configuration", config.wandb_config),
        ("model_settings", " Model Architecture", config.model_settings),
        ("dataset_config", "📁 Dataset Configuration", config.dataset_config),
        ("training_config", "🏋️ Training Configuration", config.training_config),
        ("metrics_config", "📈 Metrics Configuration", config.metrics_config),
    ]

    # Create a custom pprint formatter for better formatting
    pp = pprint.PrettyPrinter(
        indent=2,
        width=80,
        depth=None,
        compact=False,
        sort_dicts=False,  # Maintain order of fields
    )

    for section_name, display_name, section_config in config_sections:
        print(f"{indent_str}{display_name}:")

        # Convert Pydantic model to dictionary
        config_dict = section_config.model_dump()

        # Format the dictionary as a string
        config_str = pp.pformat(config_dict)

        # Add proper indentation to each line
        lines = config_str.split("\n")
        for line in lines:
            print(f"{indent_str}  {line}")

        print()  # Add spacing between sections

    print(
        f"{indent_str}═══════════════════════════════════════════════════════════════"
    )


def print_config_pretty(config: BridgeConfig, indent: int = 0) -> None:
    """Print all configuration parameters in a pretty, hierarchical format.

    Args:
        config: The BridgeConfig object to print.
        indent: Number of spaces to indent each level (default: 0).
    """
    indent_str = " " * indent

    print(
        f"{indent_str}╔══════════════════════════════════════════════════════════════"
    )
    print(f"{indent_str}║ BRIDGE MODEL CONFIGURATION")
    print(
        f"{indent_str}╚══════════════════════════════════════════════════════════════"
    )

    # WandB Configuration
    print(f"{indent_str}📊 WandB Configuration:")
    print(f"{indent_str}   ┌─ Project: {config.wandb_config.project}")
    print(f"{indent_str}   ├─ Entity: {config.wandb_config.entity}")
    print(f"{indent_str}   └─ Enabled: {config.wandb_config.is_enabled}")

    # Model Configuration
    print(f"{indent_str} Model Architecture:")
    print(f"{indent_str}   ┌─ Model Dimensions:")
    print(f"{indent_str}   │  ├─ d_model: {config.model_settings.d_model}")
    print(f"{indent_str}   │  ├─ d_embedding: {config.model_settings.d_embedding}")
    print(f"{indent_str}   │  └─ nhead: {config.model_settings.nhead}")
    print(f"{indent_str}   ├─ Layer Configuration:")
    print(
        f"{indent_str}   │  ├─ Phonology Encoder Layers: {config.model_settings.num_phon_enc_layers}"
    )
    print(
        f"{indent_str}   │  ├─ Orthography Encoder Layers: {config.model_settings.num_orth_enc_layers}"
    )
    print(
        f"{indent_str}   │  ├─ Mixing Encoder Layers: {config.model_settings.num_mixing_enc_layers}"
    )
    print(
        f"{indent_str}   │  ├─ Phonology Decoder Layers: {config.model_settings.num_phon_dec_layers}"
    )
    print(
        f"{indent_str}   │  └─ Orthography Decoder Layers: {config.model_settings.num_orth_dec_layers}"
    )
    print(f"{indent_str}   ├─ Sequence Lengths:")
    print(
        f"{indent_str}   │  ├─ Max Orthographic: {config.model_settings.max_orth_seq_len}"
    )
    print(
        f"{indent_str}   │  └─ Max Phonological: {config.model_settings.max_phon_seq_len}"
    )
    print(f"{indent_str}   ├─ Sliding Window:")
    print(f"{indent_str}   │  ├─ Window Size: {config.model_settings.window_size}")
    print(f"{indent_str}   │  ├─ Enabled: {config.model_settings.use_sliding_window}")
    print(f"{indent_str}   │  ├─ Causal: {config.model_settings.is_causal}")
    print(
        f"{indent_str}   │  └─ Ensure Contiguous: {config.model_settings.ensure_contiguous}"
    )
    print(f"{indent_str}   └─ Seed: {config.model_settings.seed}")

    # Dataset Configuration
    print(f"{indent_str}📁 Dataset Configuration:")
    print(f"{indent_str}   ┌─ File Path: {config.dataset_config.dataset_filepath}")

    print(
        f"{indent_str}   ├─ Tokenizer Cache Size: {config.dataset_config.tokenizer_cache_size}"
    )
    if config.dataset_config.custom_cmudict_path:
        print(
            f"{indent_str}   └─ Custom CMU Dict: {config.dataset_config.custom_cmudict_path}"
        )
    else:
        print(f"{indent_str}   └─ Custom CMU Dict: None")

    # Training Configuration
    print(f"{indent_str}🏋️ Training Configuration:")
    print(f"{indent_str}   ┌─ Epochs: {config.training_config.num_epochs}")
    print(f"{indent_str}   ├─ Batch Sizes:")
    print(f"{indent_str}   │  ├─ Training: {config.training_config.batch_size_train}")
    print(f"{indent_str}   │  └─ Validation: {config.training_config.batch_size_val}")
    print(f"{indent_str}   ├─ Learning Rate: {config.training_config.learning_rate}")
    print(f"{indent_str}   ├─ Weight Decay: {config.training_config.weight_decay}")
    print(
        f"{indent_str}   ├─ Training Pathway: {config.training_config.training_pathway}"
    )
    print(
        f"{indent_str}   ├─ Train/Test Split: {config.training_config.train_test_split}"
    )
    print(f"{indent_str}   ├─ Save Every: {config.training_config.save_every}")
    print(
        f"{indent_str}   ├─ Model Artifacts Dir: {config.training_config.model_artifacts_dir}"
    )
    print(f"{indent_str}   ├─ Number of Chunks: {config.training_config.num_chunks}")
    if config.training_config.checkpoint_path:
        print(
            f"{indent_str}   ├─ Checkpoint Path: {config.training_config.checkpoint_path}"
        )
    if config.training_config.test_data_path:
        print(
            f"{indent_str}   ├─ Test Data Path: {config.training_config.test_data_path}"
        )
    if config.training_config.max_nb_steps:
        print(f"{indent_str}   └─ Max Steps: {config.training_config.max_nb_steps}")
    else:
        print(f"{indent_str}   └─ Max Steps: None (use epochs)")

    # Metrics Configuration
    print(f"{indent_str}📈 Metrics Configuration:")
    print(f"{indent_str}   ┌─ Batch Metrics: {config.metrics_config.batch_metrics}")
    print(
        f"{indent_str}   ├─ Training Metrics: {config.metrics_config.training_metrics}"
    )
    print(
        f"{indent_str}   ├─ Validation Metrics: {config.metrics_config.validation_metrics}"
    )
    print(f"{indent_str}   ├─ Output Modes: {', '.join(config.metrics_config.modes)}")
    if config.metrics_config.filename:
        print(f"{indent_str}   └─ Filename: {config.metrics_config.filename}")
    else:
        print(f"{indent_str}   └─ Filename: None")

    print(
        f"{indent_str}═══════════════════════════════════════════════════════════════"
    )


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

        # Print the configuration in scalable format
        print("\n" + "=" * 60)
        print("SCALABLE CONFIGURATION FORMAT:")
        print("=" * 60)
        config.print_scalable_config()

        # Print the configuration in pretty format
        print("\n" + "=" * 60)
        print("PRETTY CONFIGURATION FORMAT:")
        print("=" * 60)
        config.print_config()

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
