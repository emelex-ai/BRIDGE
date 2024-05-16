# Wrapper around wandb to allow user use or not use it
# Author: G. Erlebacher
import wandb
from addict import Dict as AttrDict
import os
import time
from watchdog.observers import Observer
 # install watch dog with pip package
from watchdog.events import FileSystemEventHandler
import asyncio

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
        self.write_artifacts = False
        self.observer = None
        #self.is_wandb_on = False
    def set_params(self, is_wandb_on=False, is_sweep=False, config=None):
        self.is_wandb_on = is_wandb_on
        self.is_sweep = is_sweep
        self.config = AttrDict(config) if config else AttrDict({})
        print("config in set parama",self.config)
        self.write_artifacts = config['write_artifacts']#Newly added for save checkpoints
        
        self.checkpoint_dir=self.config.get('model_path')#Newly added for save checkpoints
        self.artifact_name=self.config['model_file_name']#Newly added for save checkpoints              
        print("Checkpoint directory:", self.checkpoint_dir)
        print("Artifact name:", self.artifact_name)
        self.start_monitoring()
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
    async def get_checkpoint_dir(self):
        print("inside get",self.checkpoint_dir)
        return self.checkpoint_dir

    async def get_artifact_name(self):
        return self.artifact_name

    def log(self, *kargs, **kwargs):
        if self.is_wandb_on and self.run:
            print("wandb wrapper, call self.run.log")
            self.run.log(*kargs, **kwargs)
            #time.sleep(10)
            #self.save_checkpoints_to_artifacts(self.checkpoint_dir, self.artifact_name,artifact_type="model")

    #code for the Watchdog
    def start_monitoring(self):
        """
        Idea for this function is to observe the models folder continuously and check for 
        appropriate checkpoint file creation but Observer is getting terminated after watching once.
        """
        # Convert checkpoint directory to an absolute path to avoid path issues
        self.checkpoint_dir = os.path.abspath(self.checkpoint_dir)
        print(f"Starting monitoring at: {self.checkpoint_dir}")

        # Check if directory exists
        if not os.path.exists(self.checkpoint_dir):
            print("Directory does not exist, creating directory.")
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        if not self.observer or not self.observer.is_alive():
            self.observer = Observer()
            print("observer_type",type(self.observer))

        #Invoking File system Event handler of Watchdog
        event_handler = MyHandler(self.checkpoint_dir, self.artifact_name, self, self.observer)
        self.observer.schedule(event_handler, self.checkpoint_dir, recursive=True)
        self.observer.start()
        print("Observer has been started!")

    #Commented Code for Async Implementation
    """async def some_method(self,checkpoint_dir,artifact_name,):
        # some operations
        print("inside some")
        await self.save_checkpoints_to_artifacts(checkpoint_dir, artifact_name,artifact_type="model")"""

    def save_checkpoints_to_artifacts(self, checkpoint_dir, artifact_name, artifact_type="model"):
        """
        The function is used to save checkpoints to Wandb in artfacts folder
        """
        #print("inside save checkpoints to Wandb",self.is_wandb_on,self.run,self.write_artifacts)

        if self.is_wandb_on and self.run and self.write_artifacts: #condtition to check if appropriate flags are true
            print("inside check")
            #Creating a new artifact object for organizing experiment checkpoints
            artifact = wandb.Artifact(name=artifact_name, type=artifact_type)
            artifact.add_dir(checkpoint_dir)

            for filename in os.listdir(checkpoint_dir):
                if filename.endswith('.pth'):
                    print("inside if")
                    checkpoint_files = os.path.join(checkpoint_dir, artifact_name)
                    print("Files before logging to Wandb:", checkpoint_files)
                    if(filename==artifact_name):
                           artifact.add_file(os.path.join(checkpoint_dir, artifact_name), name=artifact_name)
                    else:
                        print(f"The current file: {artifact_name} is not found in the directory or not yet created!")
                        pass
            self.run.log_artifact(artifact)
        else:
            # Do nothing if artifacts writing is disabled
            print("Write Artifacts is disabled for the Wandb ")
            pass
    
                    
                


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
        
#File Systems Event Handler for Watchdog        
class MyHandler(FileSystemEventHandler):
    def __init__(self, checkpoint_dir, artifact_name, wandb_wrapper, observer):
        self.checkpoint_dir = checkpoint_dir
        self.artifact_name = artifact_name
        self.wandb_wrapper = wandb_wrapper
        self.observer = observer
        #print("MyHandler-->",type(wandb_wrapper))

    """ When Expected checkpoint file is created this should invoke below function 
     but not triggered as expected"""
    def on_created(self, event):
        print("inside on create of My Handler class")
        if event.src_path.endswith('.pth'):
            print(f"New checkpoint file detected: {event.src_path}")
            # Trigger your save operation here
            self.wandb_wrapper.save_checkpoints_to_artifacts(self.checkpoint_dir, self.artifact_name)
            # Optionally stop the observer if you only need to act on the first file created
            #self.observer.stop()

#Implementing Async

"""async def main():
    # Create an instance of WandbWrapper
    #time.sleep(5)
    #wandb_wrapper = WandbWrapper()
    #wandb_wrapper.set_params(is_wandb_on=False, is_sweep=False, config=None)
    # Call get_checkpoint_dir asynchronously
    checkpoint_dir = WandbWrapper().get_checkpoint_dir()
    artifact_name= WandbWrapper().get_artifact_name()
    # Now you can use the checkpoint directory as needed
    print("Checkpoint directory, artifact name in async:", checkpoint_dir,artifact_name)
    
    await WandbWrapper().save_checkpoints_to_artifacts(checkpoint_dir,artifact_name,artifact_type="model")
# Run the main function using asyncio
asyncio.run(main())"""

"""checkpoint_dir = WandbWrapper().get_checkpoint_dir()
artifact_name= WandbWrapper().get_artifact_name()

WandbWrapper().save_checkpoints_to_artifacts(checkpoint_dir,artifact_name,artifact_type="model")"""



   

