from src.utils.configuration.model.model_config import ModelConfig
from src.utils.shared import BaseConfigHandler
import logging
import yaml
import os


logger = logging.getLogger(__name__)


class ModelConfigHandler(BaseConfigHandler):
    """Model specific configuration handler."""

    def _validate_config(self, config_data: dict):
        """Validate the configuration using ModelConfig."""
        return ModelConfig(**config_data)

    def _default_config(self):
        """Return a default ModelConfig instance."""
        return ModelConfig()
