import logging
from pydantic import BaseModel
import yaml

logger = logging.getLogger(__name__)


class BaseConfigHandler:
    """Base class to handle loading, validating, and adjusting configurations."""

    def __init__(self, config_filepath: str = None):
        self.config_filepath = config_filepath
        self.config = self._load_and_validate_config()

    def _load_and_validate_config(self):
        """Load and validate the configuration using Pydantic or a custom model."""
        if self.config_filepath:
            logger.info(f"Loading configuration from {self.config_filepath}")
            try:
                with open(self.config_filepath, "r") as file:
                    config_data = yaml.safe_load(file)
                    logger.info(f"Configuration file loaded successfully.")
            except FileNotFoundError:
                logger.error(f"Configuration file {self.config_filepath} not found.")
                raise
            except yaml.YAMLError as e:
                logger.error(f"Error parsing YAML configuration file: {e}")
                raise
            try:
                # Delegate to child classes for validation with specific models
                config = self._validate_config(config_data)
                logger.info(f"Configuration validated successfully.")
                return config
            except Exception as e:
                logger.error(f"Error during configuration validation: {e}")
                raise
        else:
            logger.info("No configuration file provided. Using default configuration.")
            return self._default_config()

    def _validate_config(self, config_data: dict):
        """Abstract method to be implemented in child classes to validate the config."""
        raise NotImplementedError("Child classes must implement _validate_config()")

    def _default_config(self):
        """Abstract method for returning default configurations in child classes."""
        raise NotImplementedError("Child classes must implement _default_config()")

    def get_config(self):
        """Return the validated and adjusted configuration."""
        return self.config

    def print_config(self):
        """Print the adjusted configuration."""
        print("--- Config Values ---")
        print(self.config.model_dump_json(indent=2))
