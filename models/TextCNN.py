import tensorflow as tf


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, config, num_classes):
        self.max_length = config.max_length
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        self.filter_sizes = [int(x) for x in config.filter_sizes.split(',')]
        self.num_filters = config.num_filters
        self.l2_reg_lambda = config.l2_reg_lambda

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, self.max_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_dim], -1.0, 1.0), name="W")
            self.embed_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # 이건 뭘 하는 거지
            self.embed_chars_expanded = tf.expand_dims(self.embed_chars, -1)

        # Create a convolution + maxpool layer for each filter size
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
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, shape=[-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable("W",
                                shape=[num_filters_total, num_classes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes], name="b"))
            l2_loss += tf.nn.l2_loss(W) + tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.l2_reg_lambda*l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
