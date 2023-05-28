# Wrapper around wandb to allow user use or not use it
# Author: G. Erlebacher
import wandb
from attrdict import AttrDict  # dot notation for dictionaries


class Singleton(object):
    _instance = None

    def __new__(cls, *kargs, **kwargs):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance


class MyRun(Singleton):
    """
    A proxy class to handle the `run` variable when wwanb is disabled.
    """

    def __init__(self, config=None):
        self.config = config

    def watch(self, *kargs, **kwargs):
        pass

    def log(self, *kargs, **kwargs):
        pass

    def finish(self, *kargs, **kwargs):
        pass


class MyTable:
    """
    An object to return when calling wandb.Table()
    """

    def __init__(self, config=None):
        self.config = config

    def add_data(self, *kargs, **kwargs):
        pass


class WandbWrapper(Singleton):
    """
    A wrapper around wandb to allow it to be disabled.

    While wandb.init() has a mode='disabled' option, there are problems when
    implementing sweeps. My objective was to easily handl sweep and non-sweep
    runs as easily as possible within a single code, and easily be able to turn
    wandb on and off.
    """

    def __init__(self):  # , is_wandb_on=False, is_sweep=False, config=None):
        self.my_run = MyRun()
        self.my_table = MyTable()
        self.run = None

    def set_params(self, is_wandb_on=False, is_sweep=False, config=None):
        self.is_wandb_on = is_wandb_on
        self.is_sweep = is_sweep
        self.config = AttrDict(config) if config else AttrDict({})

    def get_wandb(self):
        return self

    def init(self, *kargs, **kwargs):
        if "config" in kwargs:
            self.config = AttrDict(kwargs["config"])
        if self.is_wandb_on:
            self.run = wandb.init(*kargs, **kwargs)
            return self.run
        else:  # wandb is disabled
            self.my_run.config = self.config
        return self.my_run  # I need to return something with a config attribute

    def log(self, *kargs, **kwargs):
        if self.is_wandb_on and self.run:
            self.run.log(*kargs, **kwargs)

    def watch(self, *kargs, **kwargs):
        if self.is_wandb_on and self.run:
            self.run.watch(*kargs, **kwargs)

    def Table(self, *kargs, **kwargs):
        if self.is_wandb_on and self.run:
            return wandb.Table(*kargs, **kwargs)  # Recursion. Why?
        else:
            return self.my_table

    def sweep(self, *kargs, **kwargs):
        if len(kargs) > 0:
            self.config = AttrDict(kargs[0])
            self.run = self.my_run
            self.my_run.config = self.config
        if self.is_wandb_on and self.run and self.is_sweep:
            return wandb.sweep(*kargs, **kwargs)
        else:
            return self.my_run

    def agent(self, *kargs, **kwargs):
        if self.is_wandb_on and self.run and self.is_sweep:
            return wandb.agent(*kargs, **kwargs)

    def finish(self):
        if self.is_wandb_on and self.run:
            self.run.finish()
        else:
            return self.my_run.finish()

    def save(self):
        if self.is_wandb_on and self.run:
            self.run.save()

    def login(self):
        if self.is_wandb_on and self.run:
            wandb.login()
