import argparse
import numpy as np
import pytest
import unit_test_example as ut
import src.train
import src.main 
from src.main import *

def test_full_code():
    parser = argparse.ArgumentParser(description='Train a ConnTextUL model')
    
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size_train", type=int, default=32, help="Train batch size")
    parser.add_argument("--batch_size_val", type=int, default=32, help="Validation batch size")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--continue_training", action='store_true', help="Continue training from last checkpoint \
                        (default: new run if argument absent)")
    parser.add_argument("--d_model", type=int, default=128, help="Dimensionality of the internal model components \
                        including Embedding layer, transformer layers, \
                        and linear layers. Must be evenly divisible by nhead")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads for all attention modules. \
                        Must evenly divide d_model.")
    # Be careful when using bool arguments. Must use action='store_true', which creates an option that defaults to True 
    parser.add_argument("--wandb", action='store_true', help="Enable wandb (default: disabled if argument absent)")
    parser.add_argument("--test", action='store_true', help="Test mode: only run one epoch on a small subset of the data \
                        (default: no test if argument absent)")
    parser.add_argument("--max_nb_steps", type=int, default=-1, help="Hardcode nb steps per epoch for fast testing")
    parser.add_argument("--train_test_split", type=float, default=0.8, help="Fraction of data in the training set")

    parser.add_argument("--which_dataset", type=int, default=20, help="Choose the dataset to load")
    parser.add_argument("--sweep",type=str,  default="", help="Run a sweep from a configuration file")
    parser.add_argument("--d_embedding", type=int, default=1, help="Dimensionality of the final embedding layer.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for repeatibility.")
    parser.add_argument("--nb_samples", type=int, default=1000, help="Number of total samples from dataset.")
    args = parser.parse_args()
    metrics = main(args)
    assert 3 == 3

