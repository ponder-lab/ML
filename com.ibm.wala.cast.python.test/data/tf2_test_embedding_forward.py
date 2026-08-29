import numpy as np
import tensorflow as tf


def consume_embed(a):
    pass


def consume_rnn(b):
    pass


def consume_downstream(c):
    pass


# The real chain's head, which the earlier witnesses skipped: integer token IDs through an
# `Embedding`, then a `Bidirectional` recurrent layer, then across a layer-call boundary into a
# second user-defined layer's parameter.
class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(10, 6)
        self.rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(4, return_sequences=True)
        )

    def call(self, inputs, training):
        embed = self.embedding(inputs)
        consume_embed(embed)
        logits = self.rnn(embed, training=training)
        consume_rnn(logits)
        return logits


class Downstream(tf.keras.layers.Layer):
    def call(self, encoder_output):
        consume_downstream(encoder_output)
        return encoder_output, 1


class Net(tf.keras.Model):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = Encoder()
        self.downstream = Downstream()

    def call(self, inputs, training=True):
        logits = self.encoder(inputs, training)
        logits, _ = self.downstream(logits)
        return logits


tokens = tf.constant(np.ones((2, 5), dtype=np.int32))
out = Net()(tokens)
assert out.shape == (2, 5, 8), out.shape
assert out.dtype == tf.float32, out.dtype
