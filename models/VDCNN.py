#! /usr/bin/env python

import tensorflow as tf

class VDCNN(object):
    """
    A Very Deep CNN for text classification
    """
    def __init__(self, config, num_classes):
        self.max_length = config.max_length
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        self.