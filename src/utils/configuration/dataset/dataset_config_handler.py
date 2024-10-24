from src.utils.configuration.dataset import DatasetConfig
from src.utils.shared import BaseConfigHandler
import logging

logger = logging.getLogger(__name__)


class DatasetConfigHandler(BaseConfigHandler):
    """Dataset specific configuration handler."""

    def _validate_config(self, config_data: dict):
        """Validate the configuration using DatasetConfig."""
        return DatasetConfig(**config_data)

    def _default_config(self):
        """Return a default DatasetConfig instance."""
        return DatasetConfig()
