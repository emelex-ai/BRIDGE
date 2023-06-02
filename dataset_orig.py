from Traindata import Traindata
from torch.utils.data import Dataset
import pandas as pd
import torch
import nltk
nltk.download('cmudict')
import numpy as np
import pickle
import os

DATA_PATH = "./data/"

class CUDA_Dict(dict):

  def to(self, device):

    output = {}
    for key in self.keys():
      batches = self[key]
      if isinstance(batches, list):
        try:
          output[key] = [[val.to(device) for val in batch] for batch in batches]
        except:
          print(f"batches = {batches}")
          raise
      elif isinstance(batches, torch.Tensor):
        output[key] = batches.to(device)
      else:
        raise TypeError("Must be list or torch tensor")

    return output

class CharacterTokenizer:
    def __init__(self, list_of_characters):

        # Include custom tokens into vocabulary
        self.vocab = ['[BOS]', '[EOS]', '[CLS]', '[UNK]', '[PAD]']
        self.vocab.extend(list_of_characters)

        self.char_2_idx, self.idx_2_char = {}, {}
        for i, ch in enumerate(self.vocab):
          self.char_2_idx[ch] = i 
          self.idx_2_char[i] = ch
        # Reuse index from previous for loop to save computation
        self.size = i+1


    def __len__(self): return self.size


    def encode(self, list_of_strings):
        assert isinstance(list_of_strings, str) or (isinstance(list_of_strings, list) \
                 and all(isinstance(string, str) for string in list_of_strings))
        if isinstance(list_of_strings, str): list_of_strings = [list_of_strings]

        lengths = [len(string) for string in list_of_strings]
        max_length = max(lengths)

        # Padding function, puts beginning-of-string and end-of-string tokens
        # on beginning and end of string, after padding the original string
        # up to max length of string.
        # This function converts the output to a list rather than string.
        enc_pad = lambda string: ['[BOS]'] + list(string) + ['[EOS]'] + (max_length - len(string)) * ['[PAD]']
        dec_pad = lambda string: ['[BOS]'] + list(string) + (max_length - len(string)) * ['[PAD]']

        list_of_enc_strings = list(map(enc_pad, list_of_strings))
        list_of_dec_strings = list(map(dec_pad, list_of_strings))

        # Initiate encoder tokens tensor as tensor of zeros, 
        enc_input_ids = torch.zeros((len(list_of_enc_strings), 2 + max_length), dtype=torch.long)
        for idx, string in enumerate(list_of_enc_strings):
            for jdx, char in enumerate(string):
                enc_input_ids[idx, jdx] = self.char_2_idx.get(char, 3) # Default to [UNK]

        # Initiate decoder tokens tensor as tensor of zeros, 
        dec_input_ids = torch.zeros((len(list_of_dec_strings), 1 + max_length), dtype=torch.long)
        for idx, string in enumerate(list_of_dec_strings):
            for jdx, char in enumerate(string):
                dec_input_ids[idx, jdx] = self.char_2_idx.get(char, 3) # Default to [UNK]                

        PAD_TOKEN = self.char_2_idx['[PAD]']
        enc_pad_mask = (enc_input_ids == PAD_TOKEN)
        dec_pad_mask = (dec_input_ids == PAD_TOKEN)
        return CUDA_Dict({'enc_input_ids':enc_input_ids,
                          'dec_input_ids':dec_input_ids,
                          'enc_pad_mask':enc_pad_mask.bool(),
                          'dec_pad_mask':dec_pad_mask.bool()})
            

    def decode(self, list_of_ints):

        outputs = [''.join([self.idx_2_char.get(i) for i in ints]) for ints in list_of_ints]

        return outputs
    
# We may want to calculate all of the phonological vectors for the entire dataset ahead of time here.
# This way, we can just look up the vector instead of calculating it on the fly. That will save time
# during training.

class Phonemizer():

  def __init__(self, wordlist):

    self.PAD = 33

    self.traindata = Traindata.Traindata(wordlist, 
                          phonpath='raw/phonreps.csv',  
                          terminals=True,
                          oneletter=True,
                          verbose=False)
    traindata = self.traindata.traindata
    self.enc_inputs = {}
    self.dec_inputs = {}
    self.targets = {}
    for length in traindata.keys():
      for word_num, (phon_vec_sos, phon_vec_eos) in enumerate(zip(traindata[length]['phonSOS'], traindata[length]['phonEOS'])):
        word = traindata[length]['wordlist'][word_num]
        # The encoder receives the entire phonological vector include the BOS and EOS tokens
        self.enc_inputs[word] = [torch.tensor(np.where(vec)[0], dtype=torch.long) for vec in phon_vec_sos] + [torch.tensor([32])] # 32 is the EOS token location
        # The decoder received the entire phonological vectors including the BOS token, but not the EOS token
        self.dec_inputs[word] = [torch.tensor(np.where(vec)[0], dtype=torch.long) for vec in phon_vec_sos]
        # The target for the decoder is all phonological vectors including the EOS token, but excluding the BOS token
        self.targets[word] = phon_vec_eos

    del traindata

  def __len__(self):
    
    return 34

  def encode(self, wordlist):

    enc_input_ids = []
    dec_input_ids = []
    targets = []

    if isinstance(wordlist, list):
      max_length = 0
      for word in wordlist:
        # Make sure all words are in the phonological dictionary
        enc_input = self.enc_inputs.get(word, None)
        dec_input = self.dec_inputs.get(word, None)
        target = self.targets.get(word, None)
        # If any word is not in the dictionary, skip the batch
        if enc_input is None or dec_input is None or target is None:
          return None
        # Collect all token lists in a larger list while calculating the max length of this batch
        enc_input_ids.append(enc_input.copy())
        dec_input_ids.append(dec_input.copy())
        targets.append(torch.tensor(target.copy(), dtype=torch.long))
        # All three, enc_input, dec_input, and target should be the same length. So all share the same max_length
        # (though we subtract 1 from the decoder input and targets because the BOS/EOS tokens were removed)
        max_length = max(max_length, len(enc_input))
      # Now that we know the max length of this batch, we pad the encoder and decoder input token list with PAD tokens
      for epv, dpv in zip(enc_input_ids, dec_input_ids):
        epv.extend([torch.tensor([self.PAD])]*(max_length-len(epv)))
        dpv.extend([torch.tensor([self.PAD])]*(max_length-1-len(dpv)))
      # We then include padding, or indices in the targets to be passed to the 'ignore_index' parameter in the CrossEntropyLoss
      # Since each phonological vector is either on or off, it is a binary classification problem, so valid labels are either 0, or 1.
      # We will include labels of '2' where the padding is in the target vectors
      #print("targets = ", targets)
      for i in range(len(targets)):
        tv = targets[i]
        targets[i] = torch.concat((tv, torch.tensor([[2]*33]*(max_length-1-len(tv)), dtype=torch.long)))
        #print("len(tv) = ", len(tv))
        #tv = torch.concat((tv, torch.tensor([[2]*33]*(max_length-len(tv)))))
        #print("tv = ", tv)
        #sys.exit()
    else:
      raise TypeError('encode only accepts lists or a single string as input')

    enc_pad_mask = torch.tensor([[all(val == torch.tensor([self.PAD])) for val in token] for token in enc_input_ids])
    dec_pad_mask = torch.tensor([[all(val == torch.tensor([self.PAD])) for val in token] for token in dec_input_ids])
    #dec_pad_mask = torch.tensor([1])

    # Ensure that the number of tokens matches the number of boolean values in the mask
    assert len(enc_input_ids) == len(enc_pad_mask), f"tokens is length {len(enc_input_ids)}, enc_pad_mask is length {len(enc_pad_mask)}. They must be equal"

    # Do we need to pad the targets? We do to convert it to a tensor which is needed for the CrossEntropy criterion
    return CUDA_Dict({'enc_input_ids':enc_input_ids, 
                      'enc_pad_mask':enc_pad_mask.bool(), 
                      'dec_input_ids':dec_input_ids,
                      'dec_pad_mask':dec_pad_mask.bool(),
                      'targets':torch.stack(targets, 0)})
  
  def decode(self, tokens):
    output = torch.zeros(len(tokens), 33)
    for i, token in enumerate(tokens):
       output[i, token] = 1

    return output
  
class ConnTextULDataset(Dataset):
  """ConnTextULDataset

  Dataset of 

  For Matt's Phonoligical Feature Vectors, we will use (31, 32, 33) to represent ('[BOS]', '[EOS]', '[PAD]')

  """
  def __init__(self, test=False):

      if test:
        self.dataset = pd.read_csv(DATA_PATH+'/test.csv')
      else:
        self.dataset = pd.read_csv(DATA_PATH+'/data.csv')
      tmp_words = self.dataset['word_raw'].str.lower() # Series of all lowercased words
      if os.path.exists(DATA_PATH+'/phonology_tokenizer.pkl'):
        with open(DATA_PATH+'/phonology_tokenizer.pkl', 'rb') as f:
           self.phonology_tokenizer = pickle.load(f)
      else:
        self.phonology_tokenizer = Phonemizer(tmp_words)
        with open(DATA_PATH+'/phonology_tokenizer.pkl', 'wb') as f:
            pickle.dump(self.phonology_tokenizer, f)
      # self.listed_words = [word for word in self.words]

      # Notice I created a tokenizer in this class.
      # We can use it to tokenize word output of __getitem__ below,
      # although I haven't implemented yet.
      list_of_characters = sorted(set([c for word in tmp_words for c in word]))

      self.character_tokenizer = CharacterTokenizer(list_of_characters)

      final_words = []
      for word in tmp_words:
        if word == "" or word == None or word == []:
          continue
        if self.phonology_tokenizer.encode([word]): #check if in phoneme_dict
          final_words.append(word)

      self.words = final_words

          
  def __len__(self):
      length = len(self.words)  
      return length  

  def __getitem__(self, idx):

      string_input = self.words[idx]
      # tokenized orthographic output. character_tokenizer pads and wraps in tensor
      orth_tokenized = self.character_tokenizer.encode(string_input)
      phon_tokenized = self.phonology_tokenizer.encode(string_input)

      return {'orthography': orth_tokenized, 'phonology': phon_tokenized}
