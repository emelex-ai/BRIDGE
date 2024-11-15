import pytest
import pandas as pd
import os
import pickle
from unittest.mock import patch, mock_open
from src_legacy.dataset import ConnTextULDataset
from src_legacy.dataset import CharacterTokenizer
from src_legacy.dataset import Phonemizer
