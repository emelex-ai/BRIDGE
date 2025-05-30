from src.application.shared import BaseConfigHandler
from src.domain.datamodels import ModelConfig
import logging


logger = logging.getLogger(__name__)


class ModelConfigHandler(BaseConfigHandler):
    """Model specific configuration handler."""

    def _validate_config(self, config_data: dict):
        """Validate the configuration using ModelConfig."""
        return ModelConfig(**config_data)

    def _default_config(self):
        """Return a default ModelConfig instance."""
        return ModelConfig()
