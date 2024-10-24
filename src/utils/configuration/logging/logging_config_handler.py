import logging.config
import logging
import yaml


class LevelFilter(logging.Filter):
    """Custom filter class to allow filtering logs by specific log levels."""

    def __init__(self, level):
        super().__init__()
        self.level = level

    def filter(self, record):
        """Only allow logs of the specified level."""
        return record.levelno == self.level


class LoggingConfigHandler:
    """Class to handle logging configuration and applying custom filters."""

    def __init__(self, config_path="app/config/logging_config.yaml"):
        self.config_path = config_path

    def load_config(self):
        """Load logging configuration from a YAML file."""
        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)
            logging.config.dictConfig(config)

    def apply_filters(self):
        """Apply custom filters to the respective handlers."""
        logger = logging.getLogger()

        for handler in logger.handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                if handler.baseFilename.endswith("info.log"):
                    handler.addFilter(LevelFilter(logging.INFO))
                elif handler.baseFilename.endswith("debug.log"):
                    handler.addFilter(LevelFilter(logging.DEBUG))

    def setup_logging(self):
        """Main method to load config and apply filters."""
        self.load_config()
        self.apply_filters()
