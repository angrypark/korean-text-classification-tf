import tensorflow as tf
import argparse
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn

import tokenizers, normalizers, vectorizers
from data_helper import Preprocessor, load_data, split_data, batch_iter
from models import TextCNN


# Parameters
# ==================================================

# Task specification
args = argparse.ArgumentParser()
args.add_argument("--mode", type=str, default="train")
args.add_argument("--small", type=bool, default=False)

# Data loading parameters
args.add_argument("--train_dir", type=str, default="./data/ilbe/train.txt")
args.add_argument("--val_dir", type=str, default="./data/ilbe/test.txt")
args.add_argument("--pretrained_embed_dir", type=str, default="")
args.add_argument("--checkpoint_dir", type=str, default="")

# Model specification
args.add_argument("--model", type=str, default="TextCNN")
args.add_argument("--normalizer", type=str, default="BasicNormalizer")
args.add_argument("--tokenizer", type=str, default="JamoTokenizer")
args.add_argument("--vocab_size", type=int, default=20000)

# Model hyperparameters
args.add_argument("--embed_dim", type=int, default=128)
args.add_argument("--learning_rate", type=float, default=1e-3)
args.add_argument("--min_length", type=int, default=64)
args.add_argument("--max_length", type=int, default=512)
args.add_argument("--dropout_keep_prob", type=float, default=0.5)
args.add_argument("--l2_reg_lambda", type=float, default=0.0)

# Training parameters
args.add_argument("--batch_size", type=int, default=64)
args.add_argument("--num_epochs", type=int, default=200)
args.add_argument("--evaluate_every", type=int, default=1)
args.add_argument("--checkpoint_every", type=int, default=1)
args.add_argument("--num_checkpoints", type=int, default=30)
args.add_argument("--shuffle", type=bool, default=True)

# Misc parameters
args.add_argument("--allow_soft_replacement", type=bool, default=True)
args.add_argument("--log_device_placement", type=bool, default=False)
args.add_argument("--gpu", type=str, default="all")

config = args.parse_args()
config_str = ""
for attr, value in config.__dict__.items():
    config_str += "{}={}".format(attr, value if value else None) + " "
print(config_str)

def main():
    # Load model, normalizer and tokenizer
    # ==================================================
    Model = eval("{}.{}".format(config.model, config.model))
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

    # Build vocabulary and preprocess data
    vectorizer.build_vectorizer(train_data)
    preprocessor = Preprocessor(config, normalizer, tokenizer, vectorizer)
    train_data, val_data = preprocessor.preprocess(train_data), preprocessor.preprocess(val_data)

    # Set device setting
    device_config = tf.ConfigProto()
    device_config.allow_soft_placement = config.allow_soft_replacement
    device_config.log_device_placement = config.log_device_placement
    device_config.gpu_options.allow_growth = True

    # Training
    # ==================================================
    with tf.Session(config=device_config) as sess:
        # Create model
        num_classes = train_labels.shape[1]
        model = Model(config, num_classes)

        # Create Saver
        saver = tf.train.Saver()
        if os.path.exists(config.checkpoint_dir + "checkpoint"):
            print("Restoring variables from checkpoint : {}".format(config.checkpoint_dir))
            saver.restore(sess, tf.train.latest_checkpoint(config.checkpoint_dir))
        else:
            print("Initializing Variables")
            sess.run(tf.global_variables_initializer())
            if config.pretrained_embed_dir:
                embedding = load_pretrained_embedding(sess, config.pretrained_model_dir)
                tf.assign(model.Embedding, embedding)

        # Define training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=config.learning_rate)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity
        grad_summaries = list()
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Loss and accuracy summaries
        loss_summary = tf.summary.scalar("loss", model.loss)
        acc_summary = tf.summary.scalar("accuracy", model.accuracy)

        # Train summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Validation summaries
        val_summary_op = tf.summary.merge([loss_summary, acc_summary])
        val_summary_dir = os.path.join(out_dir, "summaries", "val")
        val_summary_writer = tf.summary.FileWriter(val_summary_dir, sess.graph)

        # Checkpoint directory
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=config.num_checkpoints)

        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {model.input_x: x_batch,
                         model.input_y: y_batch,
                         model.dropout_keep_prob: config.dropout_keep_prob}
            _, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op, model.loss,
                                                           model.accuracy],
                                                          feed_dict=feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def val_step(x_batch, y_batch):
            """
            Evaluates model on a validation set
            """
            feed_dict = {
                model.input_x: x_batch,
                model.input_y: y_batch,
                model.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run([global_step, val_summary_op, model.loss, model.accuracy],
                                                       feed_dict=feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            val_summary_writer.add_summary(summaries, step)

        # Generate batches
        batches = batch_iter(list(zip(train_data, train_labels)), config.batch_size, config.num_epochs, config.shuffle)

        # Training loop for each batch
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % config.evaluate_every == 0:
                print("\nEvaluation : ")
                val_step(val_data, val_labels)
                print("")
            if current_step % config.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))


def load_pretrained_embedding(shape, pretrained_embed_path, trainable=True):
    print("Loading pre-trained word embedding from {}".format(pretrained_embed_path))
    embedding_matrix = np.load(pretrained_embed_path)
    initializer = tf.constant_initializer(embedding_matrix.astype(np.float32))
    with tf.variable_scope("embedding"):
        embedding = tf.get_variable(name='pretrained_embedding',shape=shape, initializer=initializer, trainable=trainable)
    return embedding

if __name__== "__main__":
    main()