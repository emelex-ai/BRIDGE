from src.wandb_wrapper import WandbWrapper
from pprint import pprint
import torch  # (how does this work on github?)
import os
import shutil
import pytest
import getpass
from addict import Dict as AttrDict
from src.train_impl import get_starting_model_epoch, get_model_file_name, set_seed
import src.train_impl as train_impl
from src.train import run_code_impl
from pathlib import Path
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
    
    # function to replace get_model_file_name for testing
    def mock_file_name(model_id, epoch_nb):
        """ Same user regardless of actual user """
        file_nm = f"erlebach{model_id:05d}_chkpt{epoch_nb:03d}.pth"
        print("mock_file_name: ", file_nm)
        return file_nm

    def mock_user():
        return "erlebach"

    #monkeypatch.setattr(train_impl, "get_model_file_name", mock_file_name)
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
    #shutil.rmtree("./models_mock")

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


# How can I have arguments on my fixture? 
# Parametrized fixtures
#@pytest.fixture(scope='function')
#def metrics_gm(monkeypatch):
        #return metrics, gm

def test_runcodes(monkeypatch, model_path):
    args = "script --num_epochs 2 --test --which_dataset 100 --project proj"
    monkeypatch.setattr("sys.argv", args.split(" "))
    c = handle_arguments()
    c.model_path = model_path
    seed = c.seed
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
    # continue_training == False : create new model_id and set last_epoch_completed to 0
    # continue_training == True : 
    #  model_id == None: find highest model_id already run and extract last_epoch_completed 
    #  model_id is int > 0: find the specified model_id and extract last_epoch_completed
    model_id, last_epoch_completed = get_starting_model_epoch(
        c.model_path, model_id=None, continue_training=c.continue_training
    )
    # last_epoch_completed == 0 since continue_training is False
    # model_id == 6 since the highest model_id in models_mock is 5. 
    assert c.continue_training == False
    assert model_id == 6   
    last_epoch_completed = 0

    #print(f"==> {model_id=}, {last_epoch_completed=}")

    set_seed(c)
    print("test_restart, id(c): ", id(c)) 
    # Inside run_code_impl, there is a line: "c = run.config"

    metrics, gm = run_code_impl(run, ds, last_epoch_completed, model_id)
    c = gm.cc  # Make sure that gg.cc and c are the same.
    # It might be better if "c" were a singleton. 

    print("test_restart, id(gm): ", id(gm))
    print("test_restart, id(gm.cc): ", id(gm.cc))  # Same as c = run.config()
    print(f"1 => {model_id=}")

    # Check that a new file is created in the models_mock/ folder
    assert os.path.exists(c.model_path + "erlebach00006_chkpt002.pth")

    # Starting from the same simulation, run for 1 epoch twice
    assert c.continue_training == False
    model_id = model_id + 1  # new run 6 -> 7
    last_epoch_completed = 0
    gm.cc.epochs_completed = 0
    set_seed(c)
    gm.cc.num_epochs = 1
    pprint(gm)
    print("before run 1 epoch, model_id: ", model_id)

    # Reinialize gm
    metrics, gm = run_code_impl(run, ds, gm.cc.epochs_completed, model_id)
    print(f"Ran a first epoch: {gm.cc.epochs_completed=}")
    assert gm.cc.num_epochs == 1
    assert gm.cc.epochs_completed == 1
    print("gm.cc.num_epochs: ", gm.cc.num_epochs)
    print("after run 1 epoch, model_id: ", model_id)
    assert model_id == 7  
    assert os.path.exists(c.model_path + "erlebach00007_chkpt001.pth")

    # Run another epoch (2nd epoch)
    print(f"about to run 2nd epoch: {gm.cc.epochs_completed=}")
    gm.cc.continue_training = True
    assert gm.cc.epochs_completed == 1
    assert gm.cc.num_epochs == 1 
    metrics2 = train_impl.run_train_val_loop(gm)
    assert gm.cc.epochs_completed == 2
    print("COMPLETED 1 epoch a second time")
    assert os.path.exists(c.model_path + "erlebach00007_chkpt002.pth")  # FAILED
    # assert os.path.exists(c.model_path + "erlebach00007_chkpt003.pth")  # FILE EXISTS. WY? 
    assert model_id == 7  

    model1, _, c1, gm1 = train_impl.load_model(c.model_path, model_id=6, epochs_completed=2);
    assert gm1.model_id == 6
    model2, _, c2, gm2 = train_impl.load_model(c.model_path, model_id=7, epochs_completed=2);
    assert gm2.model_id == 7

    # Read the models saved after two epochs and compare their weights. 
    train_impl.print_weight_norms(model1, "model1 weight norms: ")  
    train_impl.print_weight_norms(model2, "model2 weight norms: ")  

    # I might only check first two matrices (to run faster)
    norm1 = torch.sqrt(sum(torch.norm(w[0], p=2) ** 2 for w in model1.parameters()))
    norm2 = torch.sqrt(sum(torch.norm(w[0], p=2) ** 2 for w in model2.parameters()))
    assert norm1 == norm2

    # Run model1 and model2 for two more epochs and check for equality
    assert gm1.cc.num_epochs == 2
    assert gm2.cc.num_epochs == 1  
    assert gm1.cc.epochs_completed == 2  
    assert gm2.cc.epochs_completed == 2 

    gm1.cc.num_epochs = 2
    gm2.cc.num_epochs = 2

    set_seed(gm1.cc)
    print("pprint gm1.cc")
    pprint(gm1.cc)
    metrics1 = train_impl.run_train_val_loop(gm1);   # Divide by zero. nb_steps == 0. len(dataset_slices) == 0. WHY?
    set_seed(gm2.cc)
    print("pprint gm2.cc")
    pprint(gm2.cc)
    metrics2 = train_impl.run_train_val_loop(gm2);
    norm1 = torch.sqrt(sum(torch.norm(w[0], p=2) ** 2 for w in model1.parameters()))
    norm2 = torch.sqrt(sum(torch.norm(w[0], p=2) ** 2 for w in model2.parameters()))
    print("---- gm1.cc")
    pprint(gm1.cc)
    print("---- gm2.cc")
    pprint(gm2.cc)
    print("----")
    print(f"{norm1=:14.7e}")
    print(f"{norm2=:14.7e}")
    assert norm1 == norm2 #norm1= 1.1248668e+01  #norm2= 1.1248734e+01 (FAILED)

    # I could continue both models for 2 epochs, and check the results. The seed has to be 
    # rest since I am restarting both from within the same program. 

    print("==================")
    print(gm1.model.state_dict())
    print("==================")
    print(gm2.model.state_dict())

    for p1, p2 in zip(gm1.model.parameters(), gm2.model.parameters()):
        assert torch.norm(p1) == torch.norm(p2)
    assert gm1.opt.state_dict() == gm2.opt.state_dict()

#----------------------------------------------------------------------
