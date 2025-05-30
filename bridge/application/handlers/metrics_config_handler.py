from bridge.application.shared import BaseConfigHandler
import logging

from bridge.domain.datamodels import MetricsConfig


logger = logging.getLogger(__name__)


class MetricsConfigHandler(BaseConfigHandler):
    """Model specific configuration handler."""

    def _validate_config(self, config_data: dict):
        """Validate the configuration using ModelConfig."""
        return MetricsConfig(**config_data)

    def _default_config(self):
        """Return a default ModelConfig instance."""
        return MetricsConfig()
