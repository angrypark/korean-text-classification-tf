import tensorflow as tf
import argparse
from datetime import datetime

from data_loader import DataGenerator
from models import TextCNN
from trainer import Trainer
from utils.dirs import create_dirs
from utils.logger import Logger

now = datetime.now()

# Parameters
# ==================================================

# Task specification
args = argparse.ArgumentParser()
args.add_argument("--name", type=str, default=now.strftime("%y%m%d_%H%M"))
args.add_argument("--mode", type=str, default="train")
args.add_argument("--config", type=str, default="")
args.add_argument("--small", type=bool, default=False)

# Data loading and saving parameters
args.add_argument("--train_dir", type=str, default="data/train.txt")
args.add_argument("--val_dir", type=str, default="data/test.txt")
args.add_argument("--pretrained_embed_dir", type=str, default="")

# Model specification
args.add_argument("--model", type=str, default="TextCNN")
args.add_argument("--normalizer", type=str, default="BasicNormalizer")
args.add_argument("--tokenizer", type=str, default="JamoTokenizer")
args.add_argument("--vocab_size", type=int, default=20000)

# Model hyperparameters
args.add_argument("--num_classes", type=int, default=3)
args.add_argument("--embed_dim", type=int, default=128)
args.add_argument("--learning_rate", type=float, default=1e-5)
args.add_argument("--min_length", type=int, default=64)
args.add_argument("--max_length", type=int, default=512)
args.add_argument("--dropout_keep_prob", type=float, default=0.9)
args.add_argument("--l2_reg_lambda", type=float, default=0.0)

# Training parameters
args.add_argument("--batch_size", type=int, default=64)
args.add_argument("--num_epochs", type=int, default=200)
args.add_argument("--evaluate_every", type=int, default=1)
args.add_argument("--checkpoint_every", type=int, default=1)
args.add_argument("--max_to_keep", type=int, default=10)
args.add_argument("--shuffle", type=bool, default=True)

# Misc parameters
args.add_argument("--allow_soft_replacement", type=bool, default=True)
args.add_argument("--log_device_placement", type=bool, default=False)
args.add_argument("--gpu", type=str, default="all")

config = args.parse_args()
config_str = " | ".join(["{}={}".format(attr.upper(), value) for attr, value in config.__dict__.items()])
print(config_str)

def main():
    # create the experiments dirs
    create_dirs(config)
    
    # create tensorflow session
    sess = tf.Session()
    
    # load data, preprocess and generate data
    data = DataGenerator(config)
    
    # create an instance of the model you want
    model = TextCNN.TextCNN(config)
    
    # create tensorboard logger
    logger = Logger(sess, config)
    
    # create trainer and pass all the previous components to it
    trainer = Trainer(sess, model, data, config, logger)
    
    # load model if exists
    model.load(sess)
    
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
