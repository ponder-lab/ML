# Test https://github.com/wala/ML/issues/988: a negative constant subscript of a tuple, `t[-k]`,
# reads element `n - k` of the tuple, so a shape argument written that way keeps its rank.
import numpy as np
import tensorflow as tf


def consume_identifiers(x):
    pass


def consume_draw(x):
    pass


def consume_zeros(x):
    pass


def consume_positive(x):
    pass


def consume_list(x):
    pass


class RemoveAccidentalNegative(tf.keras.layers.Layer):
    def call(self, logits, labels, identifiers):
        consume_identifiers(identifiers)
        return logits


shape = (2, 4)
rng = np.random.RandomState(42)
logits = rng.uniform(size=shape).astype(np.float32)
labels = np.eye(*shape).astype(np.float32)
identifiers = rng.randint(0, 3, size=shape[-1])
assert identifiers.shape == (4,) and identifiers.dtype == np.int64
RemoveAccidentalNegative()(logits, labels, identifiers)

draw = np.random.randint(0, 3, size=shape[-1])
assert draw.shape == (4,)
consume_draw(draw)

zeros = np.zeros(shape[-2])
assert zeros.shape == (2,) and zeros.dtype == np.float64
consume_zeros(zeros)

positive = np.zeros(shape[1])
assert positive.shape == (4,)
consume_positive(positive)

# A list's length can change after it is built, so its negative subscript is not read.
lengths = [2, 4]
from_list = np.zeros(lengths[-1])
assert from_list.shape == (4,)
consume_list(from_list)
