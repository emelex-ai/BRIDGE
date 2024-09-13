import pickle
import random

import torch
import numpy as np
import pandas as pd
from traindata import Traindata
from src.dataset import Phonemizer
from src.main import load_config

def phonemizer_baseline(wordlist):
    traindata = Traindata(
        wordlist,
        phonpath="data/phonreps.csv",
        terminals=True,
        oneletter=True,
        verbose=False,
    )
    traindata = traindata.traindata

    enc_inputs = {}
    dec_inputs = {}
    targets = {}
    for length in traindata.keys():
        for word_num, (phon_vec_sos, phon_vec_eos) in enumerate(
            zip(traindata[length]["phonSOS"], traindata[length]["phonEOS"])
        ):
            word = traindata[length]["wordlist"][word_num]
            # The encoder receives the entire phonological vector include the BOS and EOS tokens
            enc_inputs[word] = [
                torch.tensor(np.where(vec)[0], dtype=torch.long)
                for vec in phon_vec_sos
            ] + [
                torch.tensor([32])
            ]  # 32 is the EOS token location
            # The decoder received the entire phonological vectors including the BOS token, but not the EOS token
            dec_inputs[word] = [
                torch.tensor(np.where(vec)[0], dtype=torch.long)
                for vec in phon_vec_sos
            ]
            # The target for the decoder is all phonological vectors including the EOS token, but excluding the BOS token
            targets[word] = phon_vec_eos
    
    return enc_inputs, dec_inputs, targets

def test_phonemizer():
    # load input data
    filename = "config.yaml"
    config = load_config(filename)
    with open(config["dataset_filename"], "rb") as f:
        input_data = pickle.load(f)

    # limit to wordlist
    phonemizer = Phonemizer(config)

    # check random words
    wordlist = [w for w in input_data.keys()]
    random.seed(42)
    wordlist = random.sample(wordlist, 100)

    # original phonemizer
    enc_inputs, dec_inputs, targets = phonemizer_baseline(wordlist)

    for word in wordlist:
        shape = input_data[word]["phoneme_shape"]

        for i in range(shape[0]):
            assert torch.equal(phonemizer.enc_inputs[word][i], enc_inputs[word][i])
            assert torch.equal(phonemizer.dec_inputs[word][i], dec_inputs[word][i])
            # TODO: validate why target is a simple list
            assert torch.equal(phonemizer.targets[word][i], torch.tensor(targets[word][i], dtype=torch.int))

    pass


def test_tokenizer():
    pass
