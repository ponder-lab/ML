# An `Embedding` layer whose `output_dim` the analysis cannot read gives an output of unknown rank
# (wala/ML#1163).
import os

import tensorflow as tf
from absl import flags

flags.DEFINE_integer("d_model", 128, "The model's width.")
FLAGS = flags.FLAGS
FLAGS(["probe"])


class MultiplyConstant(tf.keras.layers.Layer):
    def __init__(self, c, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c

    def call(self, inputs):
        return inputs * self.c


def consume_literal(t):
    pass


def consume_unread(t):
    pass


def consume_flagged(t):
    pass


inp = tf.keras.layers.Input((None,), dtype="int32")
literal = tf.keras.layers.Embedding(input_dim=100, output_dim=128)(inp)
assert literal.shape.as_list() == [None, None, 128] and literal.dtype == tf.float32
consume_literal(literal)
d_model = int(os.environ.get("D_MODEL", "128"))
unread = tf.keras.layers.Embedding(input_dim=100, output_dim=d_model)(inp)
assert unread.shape.as_list() == [None, None, 128]
multiplied = MultiplyConstant(d_model)(unread)
assert multiplied.dtype == tf.float32
consume_unread(multiplied)
flagged = tf.keras.layers.Embedding(input_dim=100, output_dim=FLAGS.d_model)(inp)
assert flagged.shape.as_list() == [None, None, 128]
scaled = MultiplyConstant(FLAGS.d_model)(flagged)
assert scaled.dtype == tf.float32
consume_flagged(scaled)
