#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import math

from base.base_model import BaseModel


class VDCNN(object):
    """
    A Very Deep CNN for text classification
    """
    def __init__(self, config):
        super(VDCNN, self).__init__(config)
        self.build_model()
        self.init_saver()
        