import tensorflow as tf
import argparse
import utils
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn

from . import tokenizers, normalizers, vectorizers
from .data_helper import Preprocessor, load_data, split_data
import models


# Parameters
# ==================================================

# Task specification
args = argparse.ArgumentParser()
args.add_argument("--mode", type=str, default="train")
args.add_argument("--small", type=bool, default=False)

# Data loading parameters
args.add_argument("--train_dir", type=str, default="./data/ilbe/train.txt")
args.add_argument("--val_dir", type=str, default="./data/ilbe/test.txt")
args.add_argument("--pretrained_embed_dir", type=str, default='')

# Model specification
args.add_argument("--model", type=str, default="TextCNN")
args.add_argument("--normalizer", type=str, default="BasicNormalizer")
args.add_argument("--tokenizer", type=str, default="JamoTokenizer")
args.add_argument("--vocab_size", type=int, default=20000)

# Model hyperparameters
args.add_argument("--embed_dim", type=int, default=128)
args.add_argument("--min_length", type=int, default=64)
args.add_argument("--max_length", type=int, default=512)
args.add_argument("--filter_sizes", type=str, default="3,4,5")
args.add_argument("--num_filters", type=int, default=128)
args.add_argument("--dropout_keep_prob", type=float, default=0.5)
args.add_argument("--l2_reg_lambda", type=float, default=0.0)

# Training parameters
args.add_argument("--batch_size", type=int, default=64)
args.add_argument("--num_epochs", type=int, default=200)
args.add_argument("--evaluate_every", type=int, default=1)
args.add_argument("--checkpoint_every", type=int, default=1)

# Misc parameters
args.add_argument("--allow_soft_replacement", type=bool, default=True)
args.add_argument("--log_device_placement", type=bool, default=False)

config = args.parse_args()
logger = utils.get_logger("Text Classification")
logger.info("Arguments : {}".format(config))

# Load model, normalizer and tokenizer
# ==================================================
if config.model == 'TextCNN':
    Model = TextCNN.TextCNN
Normalizer = getattr(normalizers, config.normalizer)
Tokenizer = getattr(tokenizers, config.tokenizer)

normalizer = Normalizer(config)
tokenizer = Tokenizer(config)
vectorizer = vectorizers.Vectorizer(tokenizer, config)

# Data preparation
# ==================================================

# Load data
print("Loading data...")
train_set, val_set = load_data(config.train_dir, config.val_dir, small=config.small)
train_data, train_labels = split_data(train_set)
val_data, val_labels = split_data(val_set)

# Build vocabulary
vectorizer.build_vectorizer(train_data)
preprocessor = Preprocessor(config, normalizer, tokenizer, vectorizer)
train_data, val_data = preprocessor.preprocess(train_data), preprocessor.preprocess(val_data)

