from addict import Dict as AttrDict
from pprint import pprint
import argparse
import yaml

#----------------------------------------------------------------------
def read_args():
    parser = argparse.ArgumentParser(description="Train a ConnTextUL model")

    parser.add_argument("--batch_size_train", type=int, default=32)
    parser.add_argument("--batch_size_val", type=int, default=32)
    parser.add_argument("--continue_training",action="store_true")
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--d_embedding", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--max_nb_steps", type=int, default=-1)
    parser.add_argument("--model_path", type=str, default="./models")
    parser.add_argument("--nb_samples", type=int, default=0)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--pathway", type=str, default="op2op")
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--sweep", type=str, default="")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--train_test_split", type=float, default=0.8)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--which_dataset", type=int, default=0)
    parser.add_argument("--yaml_file", type=str, default="") # relative path to file

    args = parser.parse_args()

    assert args.which_dataset == "all" or isinstance(args.which_dataset, int)

    assert args.pathway in [
        "o2p",
        "p2o",
        "op2op",
    ], "Invalid pathway argument: must be 'o2p', 'p2o', or 'op2op'"

    return args

# ----------------------------------------------------------------------
def arg_checking():
    # Get python module to help
    pass

# ----------------------------------------------------------------------

if __name__ == '__main__':
    filenm = "root/ctul.yaml"
    with open(filenm, "r") as file:
        config = yaml.safe_load(file)

    if os.path.exists("ctul.yaml"):
        with open("ctul.yaml", "r") as file:
            config_local = yaml.safe_load(file)
            config.update(config_local)

    pprint(config)

    args = read_args()

    if args.yaml_file_path != "":
        if os.path.exists("args.yaml_file"): # And check if it is a file
            with open("args.yaml_file", "r") as file:
                config_arg = yaml.safe_load(file)
                config.update(config_arg)

    args_dct = AttrDict(vars(args))
    config.update(args_dct)

#----------------------------------------------------------------------
