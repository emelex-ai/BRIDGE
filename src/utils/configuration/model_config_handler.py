from src.utils.configuration import ModelConfig
import logging
import yaml
import os


logger = logging.getLogger(__name__)


class ModelConfigHandler:
    """Class to handle loading, validating, and adjusting the configuration."""

    def __init__(self, config_filepath: str = None):
        self.config_filepath = config_filepath
        self.config = self._load_and_validate_config()

    def _load_and_validate_config(self):
        """Load and validate the configuration using Pydantic."""
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
                config = ModelConfig(**config_data)
                logger.info(f"Configuration validated successfully.")
                return config
            except Exception as e:
                logger.error(f"Error during configuration validation: {e}")
                raise
        else:
            logger.info("No configuration file provided. Using default configuration.")
            return ModelConfig()

    def get_config(self):
        """Return the validated and adjusted configuration."""
        return self.config

    def print_config(self):
        """Print the adjusted configuration."""
        print("--- Config Values ---")
        print(self.config.model_dump_json(indent=2))
