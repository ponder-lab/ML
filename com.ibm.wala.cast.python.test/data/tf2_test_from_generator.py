# `tf.data.Dataset.from_generator` declares its element dtypes in the `output_types` tuple
# (wala/ML#776): the iterated values' dtypes are statically determined by the declaration,
# regardless of the generator body's opacity.
import numpy as np
import tensorflow as tf


def gen():
    for i in range(3):
        yield np.array([i, i + 1]), np.array([float(i)])


def consume(a):
    pass


def consume2(b):
    pass


ds = tf.data.Dataset.from_generator(gen, (tf.int32, tf.float32))
for a, b in ds:
    assert a.dtype == tf.int32
    assert b.dtype == tf.float32
    consume(a)
    consume2(b)
