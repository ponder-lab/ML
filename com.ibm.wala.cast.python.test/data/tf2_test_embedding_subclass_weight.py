# A subclass of `tf.keras.layers.Embedding` overriding `call` to gather from its own weight:
# `self.embeddings` is the `(input_dim, output_dim)` float32 weight the inherited `build` creates.
import os

import tensorflow as tf


def consume_gathered(t):
    assert t.dtype == tf.float32
    assert t.shape == (1, 3, 8), t.shape


def consume_sum(t):
    assert t.dtype == tf.float32
    assert t.shape == (1, 3, 8), t.shape


def consume_weight(t):
    assert t.dtype == tf.float32
    assert t.shape == (11, 8), t.shape


def consume_keyword_weight(t):
    assert t.dtype == tf.float32
    assert t.shape == (5, 4), t.shape


def consume_unread_weight(t):
    assert t.dtype == tf.float32
    assert t.shape[0] == 11, t.shape


class Emb(tf.keras.layers.Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        return tf.gather(self.embeddings, tf.cast(inputs, tf.int32))


ids = tf.constant([[1, 2, 3]])
emb = Emb(11, 8)
out = emb(ids)
consume_gathered(out)
consume_weight(emb.embeddings)
keyword = Emb(5, output_dim=4)
keyword(ids)
consume_keyword_weight(keyword.embeddings)
unread = Emb(11, os.cpu_count())
unread(ids)
consume_unread_weight(unread.embeddings)
plain = tf.keras.layers.Embedding(11, 8)
consume_sum(plain(ids) + out)
