from src.utils.shared import Singleton
from addict import Dict as AttrDict
import logging
import wandb

logger = logging.getLogger(__name__)


class MyRun:
    """A proxy class to handle the `run` variable when wandb is disabled."""

    def __init__(self, config=None):
        self.config = config

    def watch(self, *args, **kwargs):
        if WandbWrapper().is_wandb_on and WandbWrapper().run:
            WandbWrapper().run.watch(*args, **kwargs)

    def id(self):
        if WandbWrapper().is_wandb_on and WandbWrapper().run:
            return WandbWrapper().run.id

    def log(self, *args, **kwargs):
        pass  # Disabled

    def finish(self, *args, **kwargs):
        pass  # Disabled


class MyTable:
    """A mock object to return when calling wandb.Table() when disabled."""

    def add_data(self, *args, **kwargs):
        pass  # Disabled


class MyPlot:
    """A mock object to return when calling wandb.Plot() when disabled."""

    def histogram(self, *args, **kwargs):
        return 0  # Disabled

    def line(self, *args, **kwargs):
        return 0  # Disabled

    def scatter(self, *args, **kwargs):
        return 0  # Disabled


class WandbWrapper(Singleton):
    """
    A wrapper around wandb to allow for it to be disabled.
    """

    def __init__(self):
        self.is_wandb_on = False
        self.is_sweep = False
        self.run = None
        self.my_run = MyRun()
        self.my_table = MyTable()
        self.my_plot = MyPlot()
        self.config = AttrDict()

    def set_params(self, is_wandb_on=False, is_sweep=False, config=None):
        self.is_wandb_on = is_wandb_on
        self.is_sweep = is_sweep
        self.config = AttrDict(config) if config else AttrDict({})
        logger.info(f"Set params: is_wandb_on={self.is_wandb_on}, is_sweep={self.is_sweep}")

    def init(self, *args, **kwargs):
        if self.is_wandb_on:
            self.run = wandb.init(*args, **kwargs)
            logger.info("Initialized wandb with real run.")
            return self.run
        else:
            self.my_run.config = self.config
            logger.info("Initialized with mocked run (wandb disabled).")
            return self.my_run 

    def log(self, *args, **kwargs):
        if self.is_wandb_on and self.run:
            self.run.log(*args, **kwargs)
            logger.debug("Logged to wandb.")
        else:
            logger.debug("Logging skipped as wandb is disabled.")

    def watch(self, *args, **kwargs):
        if self.is_wandb_on and self.run:
            self.run.watch(*args, **kwargs)
            logger.debug("Watching model in wandb.")
        else:
            logger.debug("Model watch skipped as wandb is disabled.")

    def Table(self, *args, **kwargs):
        if self.is_wandb_on and self.run:
            return wandb.Table(*args, **kwargs)
        else:
            return self.my_table

    def finish(self):
        if self.is_wandb_on and self.run:
            self.run.finish()
            logger.info("Finished wandb run.")
        else:
            logger.info("Mock finish called, wandb disabled.")
