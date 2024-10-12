import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.configuration import ModelConfigHandler, LoggingConfigHandler
from addict import Dict as AttrDict
import argparse


def read_args():
    """Parse command-line arguments and return a dictionary of arguments.

    Program expexts one of the options:
        1) Nothing passed in, in which case the program uses hardcoded arguments
        2) A config file passed in, in which case the program uses the arguments in the file
    """

    parser = argparse.ArgumentParser(description="Train a ConnTextUL model")
    parser.add_argument("--config", type=str, help="Path to config file", default=None)

    args = AttrDict(vars(parser.parse_args()))
    return args


def main(config: AttrDict):
    pass


if __name__ == "__main__":
    args = read_args()

    logging_config_handler = LoggingConfigHandler()
    logging_config_handler.setup_logging()

    model_config_handler = ModelConfigHandler(config_filepath=args.config)
    model_config_handler.print_config()
    config = model_config_handler.get_config()
    main(config)
