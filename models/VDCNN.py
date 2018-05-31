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
    
    def build_model(self):
        self.max_length = self.config.max_length
        self.vocab_size = self.config.vocab_size
        self.embed_dim = self.config.embed_dim
        self.filter_sizes = 
        
        
# weights initializers
he_normal = tf.keras.initializers.he_normal()
regularizer = tf.contrib.layers.l2_regularizer(1e-4)

def convolutional_block(inputs, shortcut, num_filters, name, is_training):
    with tf.variable_scope("conv_block_" + str(num_filters) + "_" + name):
        for i in range(2):
            with tf.variable_scope("conv1d_%s" % str(i)):
                filter_shape = [3, inputs.get_shape()[2], num_filters]
                W = tf.get_variable(name='W', shape=filter_shape, 
                    initializer=he_normal,
                    regularizer=regularizer)
                inputs = tf.nn.conv1d(inputs, W, stride=1, padding="SAME")
                inputs = tf.layers.batch_normalization(inputs=inputs, momentum=0.997, epsilon=1e-5, 
                                                center=True, scale=True, training=is_training)
                inputs = tf.nn.relu(inputs)
                print("Conv1D:", inputs.get_shape())
    if shortcut is not None:
        print("-"*5)
        print("Optional Shortcut:", shortcut.get_shape())
        print("-"*5)
        return inputs + shortcut
    return inputs

# Three types of downsampling methods described by paper
def downsampling(inputs, downsampling_type, name, optional_shortcut=False, shortcut=None):
    # k-maxpooling
    if downsampling_type=='k-maxpool':
        k = math.ceil(int(inputs.get_shape()[1]) / 2)
        pool = tf.nn.top_k(tf.transpose(inputs, [0,2,1]), k=k, name=name, sorted=False)[0]
        pool = tf.transpose(pool, [0,2,1])
    # Linear
    elif downsampling_type=='linear':
        pool = tf.layers.conv1d(inputs=inputs, filters=inputs.get_shape()[2], kernel_size=3,
                            strides=2, padding='same', use_bias=False)
    # Maxpooling
    else:
        pool = tf.layers.max_pooling1d(inputs=inputs, pool_size=3, strides=2, padding='same', name=name)
    if optional_shortcut:
        shortcut = tf.layers.conv1d(inputs=shortcut, filters=shortcut.get_shape()[2], kernel_size=1,
                            strides=2, padding='same', use_bias=False)
        print("-"*5)
        print("Optional Shortcut:", shortcut.get_shape())
        print("-"*5)
        pool += shortcut
    pool = fixed_padding(inputs=pool)
    return tf.layers.conv1d(inputs=pool, filters=pool.get_shape()[2]*2, kernel_size=1,
                            strides=1, padding='valid', use_bias=False)

def fixed_padding(inputs, kernel_size=3):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [0, 0]])
    return padded_inputs