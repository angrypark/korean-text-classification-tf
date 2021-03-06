import tensorflow as tf
from base.base_model import BaseModel
import numpy as np

class TextCNN(BaseModel):
    def __init__(self, preprocessor, config):
        super(TextCNN, self).__init__(config)
        self.preprocessor = preprocessor
        self.build_model()
        self.init_saver()
    
    def build_model(self):
        # Define model parameters using config
        self.max_length = self.config.max_length
        self.vocab_size = self.config.vocab_size
        self.embed_dim = self.config.embed_dim
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 128
        self.l2_reg_lambda = self.config.l2_reg_lambda
        self.num_classes = self.config.num_classes

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, self.max_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_dim], -1.0, 1.0), name="W")
            if self.config.pretrained_embed_dir:
                print("Load pre-trained embedding from {}".format(self.config.pretrained_embed_dir))
                vocab2idx = dict()
                pretrained_embedding = np.load(self.config.pretrained_embed_dir)
                embedding = list()
                with open(self.config.vocab_list_dir, 'r') as f:
                    i = 0
                    for line in f:
                        vocab2idx[line] = i
                for idx, word in enumerate(self.preprocessor.vectorizer.idx2word):
                    try:
                        embedding.append(pretrained_embedding[vocab2idx[word]])
                    except:
                        embedding.append(np.random.uniform(-1.0,1.0,[self.embed_dim]))
                for i in range(self.vocab_size - len(embedding)):
                    embedding.append(np.random.uniform(-1.0,1.0,[self.embed_dim]))
                embedding = np.stack(embedding)
                self.W.assign(embedding)
                
            self.embed_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embed_chars_expanded = tf.expand_dims(self.embed_chars, -1)

        # Create a convolution + max pool layer for each filter size
        pooled_outputs = list()
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-{}".format(filter_size)):
                # Convolution Layer
                filter_shape = [filter_size, self.embed_dim, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embed_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Add nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h,
                                        ksize=[1, self.max_length-filter_size+1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding="VALID",
                                        name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 1)
        self.h_pool_flat = tf.reshape(self.h_pool, shape=[-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)

        # Final scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable("W",
                                shape=[num_filters_total, self.num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes], name="b"))
            l2_loss += tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            labels = tf.cast(self.input_y, tf.int64)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=labels)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda*l2_loss
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate)\
                .minimize(self.loss, global_step=self.global_step_tensor)

        # Accuracy
        with tf.name_scope("score"):
            correct_predictions = tf.equal(self.predictions, labels)
            self.score = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="score")

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
