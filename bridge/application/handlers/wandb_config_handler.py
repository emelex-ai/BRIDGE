from bridge.application.shared import BaseConfigHandler
from bridge.domain.datamodels import WandbConfig
import logging


logger = logging.getLogger(__name__)


class WandbConfigHandler(BaseConfigHandler):
    """Wandb specific configuration handler."""

    def _validate_config(self, config_data: dict):
        """Validate the configuration using WandbConfig."""
        return WandbConfig(**config_data)

    def _default_config(self):
        """Return a default WandbConfig instance."""
        return WandbConfig()
