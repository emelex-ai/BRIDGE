from attrdict import AttrDict
import argparse
import numpy as np
import pytest
from pytest import approx
import unit_test_example as ut
import src.train
#import src.main 
from src.main import main, hardcoded_args, read_args
import sys

#@pytest.fixture()
def read_my_args(dct=None):
    args_dct = AttrDict(vars(read_args()))
    test_dct = hardcoded_args()
    args_dct.update(test_dct)
    if dct != None:
        args_dct.update(dct)
    return args_dct

# next version: execute main
#def read_main_return(dct=None)
    #metrics, config = main(args_dct)

def test_main_return_test_true():
    args_dct = read_my_args({'test': True, 'num_epochs': 1, 'max_nb_steps': 1})
    return_dict = main(args_dct)
    config = return_dict.config
    assert config.test == True

def test_main_return_metrics():
    args_dct = read_my_args({'test': True})
    metrics = main(args_dct).metrics  # normally on the GPU
 
    expected_metrics_double = AttrDict({
		'train/orth_loss': 2.893103837966919,
		'train/phon_loss': 0.4451170265674591,
		'train/train_loss': 3.3382208347320557,
		'train/global_embedding_magnitude': 1.6206587553024292,
		'train/model_weights_magnitude': 10.458163261413574,
		'train/letter_wise_accuracy': 0.20000000298023224,
		'train/word_wise_accuracy': 0.0,
		'train/phon_segment_accuracy': 0.8239538073539734,
		'train/phoneme_wise_accuracy': 0.0,
		'train/phon_word_accuracy': 0.0,
		'val/orth_loss': 2.907331943511963,
		'val/phon_loss': 0.41416025161743164,
		'val/train_loss': 3.3214921951293945,
		'val/global_embedding_magnitude': 1.6206587553024292,
		'val/model_weights_magnitude': 10.458163261413574,
		'val/letter_wise_accuracy': 0.2857142984867096,
		'val/word_wise_accuracy': 0.0,
		'val/phon_segment_accuracy': 0.8528138399124146,
		'val/phoneme_wise_accuracy': 0.0,
		'val/phon_word_accuracy': 0.0,
    })

    to_del = []
    for k, v in metrics.items():
        try:
            metrics[k] = v.detach().data.item()
        except:
            to_del.append(k)

    for k, v in metrics.items():
        try:
            print("%s : %lf" % (k,v.detach().data.item()))
        except:
            pass
    print("\nreturn: metrics: ", metrics)
    to_del.extend(['train/epoch', 'train/example_ct', 'val/epoch', 'val/example_ct'])

    for k in to_del:
        try:
            del metrics[k]
        except:
            pass

    # How to check to within precision
    #assert metrics == expected_metrics_double
    assert expected_metrics_double == approx(metrics, rel=1.e-7)

def test_main_return_test_false():
    args_dct = read_my_args({'test': False, 'num_epochs': 1, 'max_nb_steps': 1})
    return_dict = main(args_dct)
    config = return_dict.config
    assert config.test == False

# function must start with test_
def test_args():
    args_dct = read_my_args({'test': True})
    assert args_dct.test == True

"""
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
"""
