# A layer whose `call` dispatches on a string `mode` argument: the embedding arm takes integer ids,
# the projection arm takes float hidden states. Each arm's helper sees only the inputs of the
# callers that select it, since the guard folds to a constant in each caller's context.
import random

import tensorflow as tf


def consume_embedding_input(x):
    return x


def consume_projection_input(x):
    return x


class SharedEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

    def build(self, input_shape):
        self.embedding_weights = self.add_weight(
            "weights", shape=[self.vocab_size, self.embedding_size], dtype="float32"
        )

    def call(self, inputs, mode="embedding"):
        if mode == "embedding":
            return self.embedding(inputs)
        elif mode == "projection":
            return self.projection(inputs)
        else:
            raise ValueError("mode {} is not valid.".format(mode))

    def embedding(self, inputs):
        consume_embedding_input(inputs)
        inputs = tf.cast(inputs, tf.int32)
        return tf.nn.embedding_lookup(self.embedding_weights, inputs)

    def projection(self, inputs):
        consume_projection_input(inputs)
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        h_flat = tf.reshape(inputs, [-1, self.embedding_size])
        logits = tf.matmul(h_flat, self.embedding_weights, transpose_b=True)
        return tf.reshape(logits, [batch_size, seq_len, self.vocab_size])


layer = SharedEmbedding(10, 8)
ids = tf.constant([[1, 2, 3], [4, 5, 6]])
hidden = layer(ids)
assert hidden.shape == (2, 3, 8) and hidden.dtype == tf.float32
logits = layer(hidden, mode="projection")
assert logits.shape == (2, 3, 10) and logits.dtype == tf.float32


def consume_undecided_projection_input(x):
    return x


class UndecidedEmbedding(tf.keras.layers.Layer):
    # A copy of the shared layer, so its helpers are reached from this control's contexts alone.
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

    def build(self, input_shape):
        self.embedding_weights = self.add_weight(
            "weights", shape=[self.vocab_size, self.embedding_size], dtype="float32"
        )

    def call(self, inputs, mode="embedding"):
        if mode == "embedding":
            return self.embedding(inputs)
        elif mode == "projection":
            return self.projection(inputs)
        else:
            raise ValueError("mode {} is not valid.".format(mode))

    def embedding(self, inputs):
        inputs = tf.cast(inputs, tf.int32)
        return tf.nn.embedding_lookup(self.embedding_weights, inputs)

    def projection(self, inputs):
        consume_undecided_projection_input(inputs)
        h_flat = tf.reshape(inputs, [-1, self.embedding_size])
        logits = tf.matmul(h_flat, self.embedding_weights, transpose_b=True)
        return tf.reshape(
            logits, [tf.shape(inputs)[0], tf.shape(inputs)[1], self.vocab_size]
        )


def call_with_mode(layer, x, flag):
    # `flag` is unresolvable, so `mode` is bound to two constants in this one context, neither arm
    # is decidably dead, and the projection helper keeps this caller's hidden states.
    m = "embedding" if flag else "projection"
    return layer(x, mode=m)


undecided = UndecidedEmbedding(10, 8)
undecided_out = call_with_mode(undecided, hidden, random.random() < 0.5)
assert (
    undecided_out.shape in ((2, 3, 8, 8), (2, 3, 10))
    and undecided_out.dtype == tf.float32
)
