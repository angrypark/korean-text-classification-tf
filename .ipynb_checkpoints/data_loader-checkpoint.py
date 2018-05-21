# -*- coding: utf-8 -*-
import random
import numpy as np
from tflearn.data_utils import pad_sequences
from collections import Counter
import os
import pickle

PAD_ID = 0
UNK_ID = 1
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

SEED = 777

def load_data_multilabel(train_data_path,
                         vocab_word2idx,
                         vocal_labe2idx,
                         sentence_len,
                         train_size=0.95):
    """
    convert data as indexes using jamo2index dicts
    :param train_data_path:
    :param vocab_word2idx:
    :param vocal_labe2idx:
    :param sentence_len:
    :param train_size:
    :return:
    """
    with open(train_data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        random.shuffle(lines, random=SEED)
        