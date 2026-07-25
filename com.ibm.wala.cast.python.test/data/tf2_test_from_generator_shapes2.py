# The keyword form of `tf2_test_from_generator_shapes.py` (wala/ML#776): `output_shapes`
# passed by keyword rather than positionally; the declaration must survive the chain the same
# way.
import numpy as np
import tensorflow as tf


def gen():
    for i in range(4):
        yield np.array([i, i + 1])


def consume(a):
    pass


ds = tf.data.Dataset.from_generator(gen, tf.int32, output_shapes=(2,))
ds = ds.shuffle(4).prefetch(tf.data.experimental.AUTOTUNE)
for a in ds:
    assert a.dtype == tf.int32
    assert a.shape == (2,)
    consume(a)
