# Wrapper around wandb to allow user use or not use it
# Author: G. Erlebacher
import wandb

class Singleton(object):
    _instance = None

    def __new__(cls, *kargs, **kwargs):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance


class MyRun(Singleton):
    def __init__(self, config=None):
        self.config = config

    def watch(self, *kargs, **kwargs):
        pass

    def log(self, *kargs, **kwargs):
        pass

    def finish(self, *kargs, **kwargs):
        pass


class MyTable:
    def __init__(self, config=None):
        self.config = config

    def add_data(self, *kargs, **kwargs):
        pass


class WandbWrapper(Singleton):
    def __init__(self, is_wandb_on=False, is_sweep=False, config=None):
        self.my_run = MyRun()
        self.my_table = MyTable()
        self.is_wandb_on = is_wandb_on
        self.is_sweep = is_sweep
        self.run = None
        self.config = config if config else {}

    def init(self, *kargs, **kwargs):
        if self.is_wandb_on:
            self.run = wandb.init(*kargs, **kwargs)
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
        if self.is_wandb_on and self.run and self.is_sweep:
            return wandb.sweep(*kargs, **kwargs)
        else:
            return 0

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
