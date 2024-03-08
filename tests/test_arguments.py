from addict import Dict as AttrDict
import argparse
import numpy as np
import pytest
from pytest import approx
import src.train
from src.main import main, hardcoded_args, read_args, handle_arguments
import sys

def test_arguments_from_cli_1(monkeypatch):
    """Test whether arguments from the command line are set up correctly."""
    # Script is a dummy variable
    monkeypatch.setattr("sys.argv", ["script", "--num_epochs", "5", "--test", "--project", "proj"])
    arg_dct = handle_arguments()
    print(arg_dct)
    assert arg_dct.test == True
    assert arg_dct.num_epochs == 5
    assert arg_dct.which_dataset == 100

def test_arguments_from_cli_2(monkeypatch):
    """Test whether arguments from the command line are set up correctly."""
    monkeypatch.setattr("sys.argv", ["script", "--test", "--which_dataset", "25", "--project", "proj"])
    arg_dct = handle_arguments()
    print(arg_dct)
    assert arg_dct.test == True
    assert arg_dct.which_dataset == 25

def test_arguments_from_cli_3(monkeypatch):
    """Test whether arguments from the command line are set up correctly."""
    monkeypatch.setattr("sys.argv", ["script", "--project", "proj"])
    arg_dct = handle_arguments()
    print(arg_dct)
    assert arg_dct.test == False
    assert arg_dct.which_dataset == 'all'

def test_arguments_from_cli_4(monkeypatch):
    """Test whether arguments from the command line are set up correctly."""
    # One can only select smaller datasets in --test mode
    monkeypatch.setattr("sys.argv", ["script", "--which_dataset", "45", "--project", "proj"])
    arg_dct = handle_arguments()
    print(arg_dct)
    assert arg_dct.test == False
    assert arg_dct.which_dataset == 'all'

