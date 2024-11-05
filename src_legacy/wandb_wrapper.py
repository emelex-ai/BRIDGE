# Wrapper around wandb to allow user use or not use it
# Author: G. Erlebacher
import wandb
from addict import Dict as AttrDict

# ----------------------------------------------------------------------
def read_wandb_history(entity, project, run):
    # Initialize the wandb API
    print("+===> enter read_wandb_history")
    api = wandb.Api()
    run_id = run.id
    run_name = run.name
    print("id, name: ", run_id, run_name)

    print(type(wandb))
    assert WandbWrapper().is_wandb_on, "Wandb must be active"
    print("wandb: ", WandbWrapper().get_wandb())
    print("wandb: ", type(WandbWrapper().get_wandb()))
    print("wandb: ", type(WandbWrapper()))
    print("wandb.run: ", type(WandbWrapper().run))

    print(f"{run_id=}")
    print(f"{wandb.run.name=}")

    # Specify the entity, project, and run ID
    print(f"{entity=}")
    print(f"{project=}")

    # Get the run object
    run = api.run(f"{entity}/{project}/{run_id}")
    print("run= ", run)
    print("run.config= ", run.config)

    """
    # Scan the history
    history = run.scan_history()
    print("history= ", history)
    print("dir(history)= ", dir(history))
    #raise "ERROR"

    # Not clear what I want to do
    # Iterate over the history records
    for record in history:
        # Access the values of the record
        print(record["step"], record["loss"])
    """
# ----------------------------------------------------------------------


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

    #def watch(self, *kargs, **kwargs):
        #pass

    def watch(self, *kargs, **kwargs):
        if WandbWrapper().is_wandb_on and WandbWrapper().run:
            #print("wandb wrapper, call self.run.watch")
            self.run.watch(*kargs, **kwargs)
    def id(self):
        if WandbWrapper().is_wandb_on and WandbWrapper().run:
            return self.run.id

    def unwatch(self, *kargs, **kwargs):
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

class MyPlot:
    """
    An object to return when calling wandb.Table()
    """

    def __init__(self, config=None):
        self.config = config

    def histogram(self, *kargs, **kwargs):
        return 0

    def line(self, *kargs, **kwargs):
        return 0

    def scatter(self, *kargs, **kwargs):
        return 0


class WandbWrapper(Singleton):
    """
    A wrapper around wandb to allow it to be disabled.

    While wandb.init() has a mode='disabled' option, there are problems when
    implementing sweeps. My objective was to easily handl sweep and non-sweep
    runs as easily as possible within a single code, and easily be able to turn
    wandb on and off.
    """

    def __init__(self):
        self.my_run = MyRun()
        self.my_table = MyTable()
        self.my_plot = MyPlot()
        self.plot = self.my_plot
        self.run = None

    def set_params(self, is_wandb_on=False, is_sweep=False, config=None):
        self.is_wandb_on = is_wandb_on
        self.is_sweep = is_sweep
        self.config = AttrDict(config) if config else AttrDict({})

    def get_wandb(self):
        return self

    def get_real_wandb(self):
        """ 
        Return the "real" wandb reference 
        Only use if wandb is enabled
        """
        if is_wandb_on:
            return wandb
        else:
            return None

    def init(self, *kargs, **kwargs):
        if "config" in kwargs:
            self.config = AttrDict(kwargs["config"])
        if self.is_wandb_on:
            self.run = wandb.init(*kargs, **kwargs)
            return self.run
        else:  # wandb is disabled
            self.my_run.config = self.config
        return self.my_run  # I need to return something with a config attribute

    def read_wandb_history(self, *kargs, **kwargs):
        if self.is_wandb_on:
            read_wandb_history(*kargs, **kwargs)

    def log(self, *kargs, **kwargs):
        if self.is_wandb_on and self.run:
            print("wandb wrapper, call self.run.log")
            self.run.log(*kargs, **kwargs)

    def watch(self, *kargs, **kwargs):
        if self.is_wandb_on and self.run:
            #print("wandb wrapper, call self.run.watch")
            self.run.watch(*kargs, **kwargs)

    def Table(self, *kargs, **kwargs):
        if self.is_wandb_on and self.run:
            return wandb.Table(*kargs, **kwargs) 
        else:
            return self.my_table

    def plot_table(self, *kargs, **kwargs):
        if self.is_wandb_on and self.run:
            return wandb.plot_table(*kargs, **kwargs)  
        else:
            return self.my_plot  # could return anything I think

    def finish(self):
        if self.is_wandb_on and self.run:
            self.run.finish()
        else:
            return self.my_run.finish()

    def login(self):
        if self.is_wandb_on and self.run:
            wandb.login()

    def save(self):
        if self.is_wandb_on and self.run:
            self.run.save()

    def is_watching(self, model):
        if self.is_wandb_on and self.run:
            return wandb.is_watching(model)
        else:
            return None

    def watch(self, model):
        if self.is_wandb_on and self.run:
            return wandb.watch(model)
        else :
            return None

    def unwatch(self, model):
        if self.is_wandb_on and self.run:
            return wandb.unwatch(model)
        else :
            return None

    #def remove_hook(self, model):
        #if self.is_wandb_on and self.run:
            #wandb.remove_hook(model)

    # Required because the wandb object in the code is a WandbWrapper object 

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
