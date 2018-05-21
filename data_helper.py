# -*- coding: utf-8 -*-
import numpy as np
from tflearn.data_utils import pad_sequences

PAD_ID = 0
UNK_ID = 1
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def load_data(train_dir, val_dir, small=False):
    """
    load data, shuffle, and train/validate split.
    :param val_size: portion of validataion data.
    :param shuffle: if True, shuffles data with predefined random seed.
    :small: only for test. If True, it makes a very small dataset to make sure if all works do properly.
    """
    with open(train_dir, 'r') as f:
        train_data = [line.strip() for line in f.readlines()]
        if small:
            train_data = train_data[:500]
    with open(val_dir, 'r') as f:
        val_data = [line.strip() for line in f.readlines()]
        if small:
            val_data = val_data[:50]
    return train_data, val_data

def split_data(data):
    lines = [line.split('\t')[0] for line in data]
    labels = [line.split('\t')[1] for line in data]
    return lines, labels

def batch_iter(data, batch_size, num_epochs, shuffle=False):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size-1)/batch_size)+1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutate(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_idx = batch_num*batch_size
            end_idx = min((batch_num+1)*batch_size, data_size)
            yield shuffled_data[start_idx:end_idx]


class Preprocessor:
    def __init__(self, config, normalizer, tokenizer, vectorizer, feature_extractors=list()):
        self.min_length = config.min_length
        self.max_length = config.max_length
        self.normalizer = normalizer
        self.tokenizer = tokenizer
        self.feature_extractors = feature_extractors
        self.vectorizer = vectorizer

    def _preprocess(self, lines):
        normalized_lines = [self.normalizer.normalize(line) for line in lines]
        tokenized_lines = [self.tokenizer.tokenize(line) for line in normalized_lines]
        extracted_features = dict()
        for feature_name, feature_extractor in self.feature_extractors:
            extracted_features[feature_name] = feature_extractor.extract_feature(
                [(line, tokens) for line, tokens in zip(lines, tokenized_lines)])
        indexed_tokens = [[self.vectorizer.indexer(token) for token in tokens] for tokens in tokenized_lines]
        return indexed_tokens, extracted_features

    def preprocess(self, lines):
        indexed_tokens, _ = self._preprocess(lines)
        padded_tokens = pad_sequences(indexed_tokens, maxlen=self.max_length)
        return padded_tokens


# class Dataset:
#     def __init__(self, data):
