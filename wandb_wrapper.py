# Wrapper around wandb to allow user use or not use it
# Author: G. Erlebacher
import wandb

class MyRun:
    def __init__(self, config=None):
        self.config = config
    def watch(self, *kargs, **kwargs):
        pass
    def log(self, *kargs, **kwargs):
        pass

class MyTable:
    def __init__(self, config=None):
        self.config = config
    def add_data(self, *kargs, **kwargs):
        pass


class WandbWrapper:
    def __init__(self, is_wandb_on, config=None):
        self.my_run = MyRun()
        self.my_table = MyTable()
        self.is_wandb_on = is_wandb_on
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
            self.run.watch(*kwargs, **kwargs)

    def Table(self, *kargs, **kwargs):
        if self.is_wandb_on and self.run:
            return self.run.Table(*kwargs, **kwargs)
        else:
            return self.my_table


    def finish(self):
        if self.is_wandb_on and self.run:
            self.run.finish()

    def save(self):
        if self.is_wandb_on and self.run:
            self.run.save()

    def login(self):
        if self.is_wandb_on and self.run:
            wandb.login()

