# %%

import numpy as np
import json
import nltk
from copy import deepcopy as cp
import string

from .utilities import phontable, phonemedict, represent, n_syllables, reconstruct, hot_nodes, key





# %%
class Traindata():

    """This class delivers representations for words to be used in ANNs.
    It is similar to CMUdata, stored elsewhere, but different because of how
    opinionated it is about the structure of its outputs and how to clean its
    inputs (words).
    ...

    traindata : dict
        The payload of the class. The traindata dictionary contains keys
        for the phonological length (plus the terminal segment if terminals=True)
        for all words provided at class call. Within phonological lengths,
        the suboridinate dictionary contains values for orthography,
        phonology, frequency, and the list of words (strings) for that set
        of words within that phonological length. If terminals=True, then
        for phonology key-values are provided with the word-initial segment
        ("SOS" for "start-of-string") and the word-final ("EOS" for "end-of-string)
        termnal segments. These terminal segments are provided in particular
        for temporal/ sequence-based ANN implementation where word boundaries
        need to be represented explicitly.

    cmudict : dict
        A dictionary which contains the string form of words as keys and
        the CMUdict encoding (list of strings) as values.

    pool : list
        The string form of the words provided with cleaning applied as specified
        in the arguments at class call.
    
    phonpath : str
        Path to the data specifying the binary representations for all the phonemes
        used to compile training data.

    phontable : pandas.DataFrame
        Compiled table of phonemes and their binary representations. Rows are
        phonemes and columns are distinctive features.

    phonreps : dict
        The binary phoneme representations in dictionary form. Keys are phonemes
        (identified as strings) and values are the binary featural representations
        of each phoneme (as a list).
        
    orthreps : dict
        The binary or scalar featural representations of orthographic segments.
        Keys are letters and values are the featural representations. If onehot
        is set to True when the class Representations() is called, the value
        is a scalar indicating which node should be hot.
    
    outliers : list
        Words identified as outliers based on what is provided to the outliers
        argument when Representations() is called. Their status as outliers
        is arbitrary and selected by the user.
    
    excluded : list
        Words removed from the training data compiled based on other
        arguments specified when Representations() is called (e.g., words
        that have more than maxsyll syllables).

    cmudictSOS : dict
        A dictionary specifying the phoneme coding for words that contains
        the word-initial (i.e. "SOS") terminal segment.

    cmudictEOS : dict
        A dictionary specifying the phoneme coding for words that contains
        the word-final (i.e. "EOS") terminal segment.

    pool_with_pad : dict
        The words identified for compiling in traindata as keys with the
        padded form of the word as the value. Default pad character is "_".
    
    self.orthlengths : dict
        A dictionary which has as keys the string form of each word, whose
        value is the orthographic length of the word (number of letters
        without consideration for the pad).

    self.orthforms : dict
        A dictionary which has as keys the string form of each word, whose
        value is a binary orthographic representation of that word in list form.

    orthforms_padded : dict
        A dictionary which has as keys the string form of each word, whose
        value is a padded binary orthographic representation of that word in list form.

    self.phonforms
        A dictionary which has as keys the string form of each word, whose
        value is a binary phonological representation of that word in list form.

    """




    def __init__(self, words, outliers=None, cmudict_supplement=None, phonpath=None, oneletter=False, maxorth=None, maxphon=None, minorth=None, minphon=None, maxsyll=None, onehot=True, orthpad=9, phonpad=9, terminals=False, phon_index=0, justify='left', punctuation=False, numerals=False, tolower=True, frequency=None, test_reps=True, verbose=True):
       
        """Initialize Representations with values that specify representations over words.
        
        Parameters
        ----------
        words : list
            A list of ortohgraphic wordforms to be encoded into orthographic
            and phonological representations.

        outliers : list or None
            A list of words to be excluded from representations of words, or None
            if no excluding is required. (default None)
        
        cmudict_supplement : str or None
            A path to a json file to be used to supplement phonological
            transcriptions contained in cmudict. Keys in the dict object
            represented in the json file should match words provided.

        phonpath : str
            Path to table of machine readable representations of phonemes
            and their string identifiers. Defaults to a repo containing these.

        phonreps : dict
            A dictionary specifying the feature representations for all phonemes
            used in constructing representations for words. The keys of this dictionary
            are phonemes, and the value for a given key is the featural representation
            for that phoneme (provided as a list).

        orthreps : dict
            A dictionary specifying the feature representations for all letters
            used in constructing representations for words. The keys of this dictionary
            are letters, and the value for a given key is the featural representation
            for that letter (provided as a list).

        oneletter : bool
            Whether to exclude words that are one letter long or not. Note the partial
            redundancy with the minorth parameter. (Default is True)
        
        maxorth : int or None
            The maximum length of the orthographic wordforms to populate the pool
            for representations. This value is calculated inclusively. (Default is None)
        
        maxphon : int or None
            The maximum length of the phonological wordforms to populate the pool
            for representations. This value is calculated inclusively. (Default is None)

        minorth : int or None
            The minimum length of the orthographic wordforms to populate the pool
            for representations. This value is calculated inclusively. (Default is None)
        
        minphon : int or None
            The minimum length of the phonological wordforms to populate the pool
            for representations. This value is calculated inclusively. (Default is None)

        maxsyll : int
            The maximum number of syllables allowed for output words. Defaults
            to None, which let's all words provided to be considered for representation
            regardles of their syllabic length. Syllabic length is calculated as the
            total number of nuclei given by the CMUdict phonological coding.
        
        onehot : bool
            Specify orthographic representations with onehot codings. (Default is True)
        
        orthpad : int or None
            The value to supply as an orthographic pad. (Default is 9)

        phonpad : int or None
            The value to supply as a phonological pad. (Default is 9)
        
        terminals : bool
            Include terminal markers in phonological representations or not. (default is False)
        
        phon_index : int or string
            The index to use for specifying which phonological representation
            from cmudict to pass for conversion to binary phonological representation.

        justify : str
            How to justify the patterns output. This specification is applied to
            all patterns produced (orthography, phonology, and if eos is
            set to True, both input and output phonology). Note that a left justification
            means that the pad is placed on the right side of the representations,
            and vice versa for right justification. (Default is left.)

        punctuation : bool
            Specifies whether (False) or not (True) punctuation should be removed. 
            Default is False, which results in punctuation being removed. Note that
            if punctuation is left (i.e., True value provided), representations for 
            punctuation need to be specified.
        
        numerals : bool
            Should numerals be present (False) or not (True). Default is False,
            which results in numerals being removed from input words.
        
        tolower : bool
            Should the words provided be converted to lowercase (True) or not (False),
            before being represented as orthographic and phonological codes. The default
            is True, resulting in all words being converted to lowercase.
        
        frequency : dict
            A dictionary of word frequencies where the keys match the words provided
            and the value of each key is that word's raw frequency. Default is None.

        test_reps : bool
            Should tests be performed on the representations to make sure they have
            expected structure. Default is True.

        """

        pool = list(set(words))

        cmudict = {word: phonforms[phon_index] for word, phonforms in nltk.corpus.cmudict.dict().items() if word in pool}

        if outliers is not None:
            if type(outliers) == str:
                outliers = [outliers]
            excluded = {word:"identified as outlier at class call" for word in outliers}
            pool = [word for word in pool if word not in outliers]
        else:
            excluded = {}

        if cmudict_supplement is not None:
            with open(cmudict_supplement, 'r') as f:
                supp = json.load(f)
            for word, phonforms in supp.items():
                if word in pool:
                    cmudict[word] = phonforms[phon_index]

        notin_cmu = [word for word in pool if word not in cmudict.keys()]
        pool = [word for word in pool if word not in notin_cmu]
        for word in notin_cmu:
            excluded[word] = "missing from cmudict"
            print(word, 'removed from pool because it is missing in cmudict')

        if not oneletter:
            for word in pool:
                if len(word) == 1:
                    pool.remove(word)
                    print(word, 'removed from pool because it has one letter')
                    excluded[word] = "one letter word"

        if maxorth is not None:
            toomanyletters = [word for word in pool if len(word) > maxorth]
            pool = [word for word in pool if word not in toomanyletters]
            for word in toomanyletters:
                excluded[word] = "too many letters"
                print(word, 'removed from pool because it has too many letters')

        if maxphon is not None:
            toomanyphones = [word for word in pool if len(cmudict[word]) > maxphon]
            pool = [word for word in pool if word not in toomanyphones]
            for word in toomanyphones:
                excluded[word] = "too many phonemes"
                print(word, 'removed from pool because it has too many phonemes')


        if minorth is not None:
            toofewletters = [word for word in pool if len(word) < minorth]
            pool = [word for word in pool if word not in toofewletters]
            for word in toofewletters:
                excluded[word] = "too few letters"
                print(word, 'removed from pool because it has too few letters')

        if minphon is not None:
            toofewphones = [word for word in pool if len(cmudict[word]) < minphon]
            pool = [word for word in pool if word not in toofewphones]
            for word in toofewphones:
                excluded[word] = "too few phonemes"
                print(word, 'removed from pool because it has too few phonemes')

        if maxsyll is not None:
            toomanysyllables = [word for word in pool if n_syllables(cmudict[word]) > maxsyll]
            pool = [word for word in pool if word not in toomanysyllables]
            for word in toomanysyllables:
                excluded[word] = "too many syllables"
                print(word, 'removed from pool because it has too many syllables')

        if not punctuation:
            punct = string.punctuation
            has_punct = [word for word in pool for ch in word if ch in punct]
            pool = [word for word in pool if word not in has_punct]
            for word in has_punct:
                excluded[word] = "puctuation present"
                print(word, 'removed because punctuation is present')

        if not numerals:
            has_numerals = [word for word in pool if any(ch.isdigit() for ch in word)]
            pool = [word for word in pool if word not in has_numerals]
            for word in has_numerals:
                excluded[word] = 'contains numerals'
                print(word, 'removed because it contains numerals')

        if tolower:
            pool = [word.lower() for word in pool]
        
        # from here the words in cmudict and pool are set
        self.cmudict = {word: phonform for word, phonform in cmudict.items() if word in pool}
        self.pool = pool

        if phonpath is None:
            # use a backup if not specified.
            phonpath = 'https://raw.githubusercontent.com/MCooperBorkenhagen/ARPAbet/master/phonreps.csv'

        self.phonpath = phonpath
        self.phontable = phontable(phonpath)
        self.phonreps = phonemedict(phonpath, terminals=terminals)
        self.phonreps_hot_nodes = {phoneme : hot_nodes(rep) for phoneme, rep in self.phonreps.items()}



        if onehot:
            orthpath = 'raw/orthreps_onehot.json'
        elif not onehot:
            orthpath = 'raw/orthreps.json'

        with open(orthpath, 'r') as f:
            orthreps = json.load(f)
        
        self.orthreps = orthreps
        self.orthreps_hot_nodes = {letter : hot_nodes(rep) for letter, rep in self.orthreps.items()}
        self.outliers = outliers
        self.excluded = excluded


        # if onehot is selected for orth, make sure that the orthographic pad is changed to 0
        # this is important because in onehot encodings, a pad of 9 corresponds to the letter <i>
        if onehot and orthpad == 9:
            orthpad = 0
            print('orthpad changed to 0 because onehot encodings were selected for orthography')

        # generate the padded version:
        if phonpad != 0:
            padrep = []
            for f in self.phonreps['_']:
                padrep.append(phonpad)
            self.phonreps['_'] = padrep

        if orthpad != 0:
            padrep = []
            for f in self.orthreps['_']:
                padrep.append(orthpad)
            self.orthreps['_'] = padrep

        # test that all the phonological vectors are the same length        
        veclengths = set([len(v) for v in self.phonreps.values()])
        assert(len(veclengths) == 1), 'Phonological feature vectors across phonreps have different lengths.'
        # derive the length of a single phoneme
        self.phoneme_length = next(iter(veclengths))


        if not onehot:
            veclengths = set([len(v) for v in self.orthreps.values()])
            assert(len(veclengths) == 1), 'Orthographic feature vectors across phonreps have different lengths.'

        if terminals:
            cmudictSOS = {}
            cmudictEOS = {}
            for word in self.pool:
                sos = cp(self.cmudict[word])
                eos = cp(self.cmudict[word])

                sos.insert(0, '#')
                eos.append('%')

                cmudictSOS[word] = sos
                cmudictEOS[word] = eos

            self.cmudictSOS = cmudictSOS
            self.cmudictEOS = cmudictEOS

        self.orthforms = {word: represent(word, self.orthreps) for word in self.pool}
        self.orthlengths = {word: len(orthform) for word, orthform in self.orthforms.items()}

        if terminals:
            self.phonformsSOS = {word: represent(self.cmudictSOS[word], self.phonreps) for word in self.pool}
            self.phonformsEOS = {word: represent(self.cmudictEOS[word], self.phonreps) for word in self.pool}
        elif not terminals:
            self.phonforms = {word: represent(self.cmudict[word], self.phonreps) for word in self.pool}
        
        if terminals:
            self.phonlengths = {word: len(phonform) for word, phonform in self.phonformsEOS.items()} # SOS could be used too
        elif not terminals:
            self.phonlengths = {word: len(phonform) for word, phonform in self.phonforms.items()}

        # maximum phonological length and orthographic length are derived if they aren't specified at class call
        if maxorth is None:
            self.maxorth = max(self.orthlengths.values())
        else:
            self.maxorth = maxorth
        
        if maxphon is None:
            self.maxphon = max(self.phonlengths.values())
        else:
            self.maxphon = maxphon

        self.pool_with_pad = {}

        for word in self.pool:
            # pad lengths for orthographic inputs (only):
            opl = self.maxorth-self.orthlengths[word]
            orthpad = ''.join(['_' for p in range(opl)])

            if justify == 'left':
                self.pool_with_pad[word] = word+orthpad
            elif justify == 'right':
                self.pool_with_pad[word] = orthpad+word

        
        self.orthforms_padded = {word: represent(orthform, self.orthreps) for word, orthform in self.pool_with_pad.items()}

        self.traindata = {}
        for length in set(self.phonlengths.values()):
            traindata_ = {}
            wordlist = [word for word, phonlength in self.phonlengths.items() if phonlength == length]

            if terminals:
                phonarraySOS = [self.phonformsSOS[word] for word in wordlist]
                phonarrayEOS = [self.phonformsEOS[word] for word in wordlist]
                traindata_['phonSOS'] = np.array(phonarraySOS)
                traindata_['phonEOS'] = np.array(phonarrayEOS)
            elif not terminals:
                phonarray = [self.phonforms[word] for word in wordlist]
                traindata_['phon'] = np.array(phonarray)

            ortharray = [self.orthforms_padded[word] for word in wordlist]
            traindata_['orth'] = np.array(ortharray)
            traindata_['wordlist'] = wordlist
                
            ##################
            # FREQUENCY DATA #
            ##################
            if frequency is not None:
                frequencies = []
                for word in wordlist:
                    frequencies.append(frequency[word])
                traindata_['frequency'] = np.array(frequencies)

            self.traindata[length] = traindata_





        #########
        # TESTS #
        #########
        if test_reps:
            for length, d in self.traindata.items():
                assert reconstruct(d['orth'], [self.pool_with_pad[word] for word in d['wordlist']], repdict=self.orthreps, join=True), 'The padded orthographic representations do not match their string representations'
                if terminals:
                    assert reconstruct(d['phonSOS'], [self.cmudictSOS[word] for word in d['wordlist']], repdict=self.phonreps, join=False), 'SOS phonological representations do not match their string representations'
                    assert reconstruct(d['phonEOS'], [self.cmudictEOS[word] for word in d['wordlist']], repdict=self.phonreps, join=False), 'EOS phonological representations do not match their string representations'
                elif not terminals:
                    assert reconstruct(d['phon'], [self.cmudict[word] for word in d['wordlist']], repdict=self.phonreps, join=False), 'Phonological representations do not match their string representations'
                if verbose:
                    print('Words of phonological length', length, 'pass reconstruction test')


        # check that all the phonemes in words in pool are represented in phonreps:
        phones = [phone for word in pool for phone in self.cmudict[word]]
        assert set(phones).issubset(self.phonreps.keys()), 'Some phonemes are missing in your phonreps'

        # check that all the letters in pool are represented in orthreps:
        letters = []
        for word in self.pool:
            for l in word:
                letters.append(l)
        assert set(letters).issubset(self.orthreps.keys()), 'there are missing binary representations for letters in the set of words'

        # perform a test on all letters, making sure that we have a binary representation for it
        if terminals:
            assert set(self.orthforms.keys()) == set(self.phonformsSOS.keys()) == set(self.phonformsEOS.keys()), 'The keys in your orth and phon dictionaries do not match'
        elif not terminals:
            assert set(self.orthforms.keys()) == set(self.phonforms.keys()), 'The keys in your orth and phon dictionaries do not match'

        print('Representations initialized. Done.')

    def convert_numeric_prediction(self, prediction, phonology=True, hot_nodes=True):

        """Convert a numeric prediction of orthographic or phonological output to a human-readable format.
        
        Parameters
        ----------
        prediction : list or array
            A numeric prediction from a model that has learned from this traindata. This
            prediction is specified as a list or array, whose elements are lists or arrays.
            Each element corresponds to a phoneme (if phonology=True) or letter (if False).

        phonology : bool
            Is the prediction a phonological one (True) or orthographic one (False). This
            argument specifies which phonreps to reference internal to Traindata. If
            phonology is set to True, the phonreps of Traindata are referenced, and if False
            the orthreps of Traindata are referenced (see hot_nodes parameter for more on this).

        hot_nodes : bool
            Is the prediction provided with specifications of the hot nodes over feature
            representations (hot_nodes=True) or with the true distributed (binary) feature
            representations (hot_nodes=False). If hot_nodes is set to True, the representations
            (attributes) in Traindata are specified with the _hot_nodes suffix
            (i.e., phonreps_hot_nodes or orthreps_hot_nodes). If hot_nodes is set to False
            then the phonreps (if phonology=True) or orthreps (if phonology=False) are referenced.
            Defaults to True for readability of predictions.

        Returns
        -------
        list or str
            A human-readable version of the prediction is provided. If phonological, the return
            object is a list. If orthographic, the return object is a string.

        """

        if phonology:
            if hot_nodes:
                return [key(self.phonreps_hot_nodes, rep) for rep in prediction]
            elif not hot_nodes:
                return [key(self.phonreps, rep) for rep in prediction]
        elif not phonology:
            if hot_nodes:
                return ''.join([key(self.orthreps_hot_nodes, rep) for rep in prediction])
            elif not hot_nodes:
                return ''.join([key(self.orthreps, rep) for rep in prediction])

if __name__ == "__main__":
    
    print("Import module and provide words to represent. Default words not provided.")
