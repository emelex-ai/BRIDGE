from bridge.application.shared import BaseConfigHandler
from bridge.domain.datamodels import TrainingConfig
import logging

logger = logging.getLogger(__name__)


class TrainingConfigHandler(BaseConfigHandler):
    """Training specific configuration handler."""

    def _validate_config(self, config_data: dict):
        """Validate the configuration using TrainingConfig."""
        return TrainingConfig(**config_data)

    def _default_config(self):
        """Return a default TrainingConfig instance."""
        return TrainingConfig()
