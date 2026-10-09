# A subclass of `tf.keras.layers.Embedding` overriding `call` to gather from its own weight:
# `self.embeddings` is the `(input_dim, output_dim)` float32 weight the inherited `build` creates.
import tensorflow as tf


def consume_gathered(t):
    assert t.dtype == tf.float32
    assert t.shape == (1, 3, 8), t.shape


def consume_sum(t):
    assert t.dtype == tf.float32
    assert t.shape == (1, 3, 8), t.shape


class Emb(tf.keras.layers.Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs):
        return tf.gather(self.embeddings, tf.cast(inputs, tf.int32))


ids = tf.constant([[1, 2, 3]])
emb = Emb(11, 8)
out = emb(ids)
consume_gathered(out)
plain = tf.keras.layers.Embedding(11, 8)
consume_sum(plain(ids) + out)
