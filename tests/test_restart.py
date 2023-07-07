from src.wandb_wrapper import WandbWrapper
import os
import shutil
import pytest
import getpass
from attrdict import AttrDict
from src.train_impl import get_starting_model_epoch, get_model_file_name
import src.train_impl as train_impl
from src.train import run_code_impl
from pathlib import Path
from pprint import pprint
from src.main import handle_arguments
from src.dataset import ConnTextULDataset

"""
Six tests: continue_training == True/False && model_id == None/int/int
Create a fake restart file: erlebach00003_chkpt000.pth (with the user name of the person conducting the test)
   (This might not work since on github the user is not clear)
def get_starting_model_epoch(model_path, model_id=None, continue_training=False):
"""
@pytest.fixture(scope='function')
def model_path(monkeypatch):
    print("==> enter fixture")
    def mock_file_name(model_id, epoch_nb):
        """ Same user regardless of actual user """
        return f"erlebach{model_id:05d}_chkpt{epoch_nb:03d}"

    def mock_user():
        return "erlebach"

    monkeypatch.setattr(train_impl, "get_model_file_name", mock_file_name)
    monkeypatch.setattr(getpass, "getuser", mock_user)
    folder_mock = "./models_mock"
    try:
        os.mkdir(folder_mock)
    except:
        pass
    file1 = folder_mock+"/erlebach00004_chkpt005.pth"
    file2 = folder_mock+"/erlebach00005_chkpt012.pth"
    Path(file1).touch()
    Path(file2).touch()
    yield  "./models_mock/"

    # Tear down the fixture
    shutil.rmtree("./models_mock")
#--------------------------------------------------------

def test_restart0(model_path):
    new_model_id, epochs_completed = get_starting_model_epoch(model_path, model_id=None, continue_training=False)
    assert new_model_id == 6    # since continue_training == None

def test_restart1(model_path):
    # file with model_id==4 exists. 
    new_model_id, epochs_completed = get_starting_model_epoch(model_path, model_id=4, continue_training=False)
    assert new_model_id == 6    # since continue_training == None

def test_restart2(model_path):
    # overwrite model_id with None when not a continuation run
    new_model_id, epochs_completed = get_starting_model_epoch(model_path, model_id=6, continue_training=False)
    # highest model_id in model_path is 4, so continue with model_id=4
    assert new_model_id == 6 and epochs_completed == 0  # ERROR

def test_restart3(model_path):
    # file with model_id==5 does not exist
    new_model_id, epochs_completed = get_starting_model_epoch(model_path, model_id=None, continue_training=True)
    assert new_model_id == 5 and epochs_completed == 12

def test_restart4(model_path):
    # file with model_id==5 does not exist
    new_model_id, epochs_completed = get_starting_model_epoch(model_path, model_id=4, continue_training=True)
    assert new_model_id == 4 and epochs_completed == 5    # since continue_training == None

def test_restart5(model_path):
    # file with model_id==5 does not exist
    new_model_id, epochs_completed = get_starting_model_epoch(model_path, model_id=6, continue_training=True)
    assert epochs_completed == -1  # since continue_training == None


def test_runcodes(monkeypatch, model_path):
    args = "script --num_epochs 5 --test --which_dataset 100 --project proj"
    monkeypatch.setattr("sys.argv", args.split(" "))
    c = handle_arguments()
    c.model_path = model_path
    pprint(c)

    wandb = WandbWrapper()
    wandb.set_params(config=c, is_sweep=False, is_wandb_on=False)
    wandb.login()

    run = wandb.init(
        name="fake_run", 
        entity="my_entity",  
        project=c.project,
        config=c,
    )

    ds = ConnTextULDataset(
        c, test=c.test, which_dataset=c.which_dataset, nb_rows=c.nb_samples
    )
    model_id, epoch_num = get_starting_model_epoch(
        c.model_path, model_id=None, continue_training=c.continue_training
    )
    print(f"{model_id=}, {epoch_num=}")
    metrics, gm = run_code_impl(run, ds, epoch_num, model_id)

#----------------------------------------------------------------------
