# `tf.data.Dataset.from_generator` with BOTH axes declared (wala/ML#776): `output_types`
# fixes the element dtype and `output_shapes` the element shape, so the declaration must
# survive the pass-through combinator chain to the iterated value on both axes.
import numpy as np
import tensorflow as tf


def gen():
    for i in range(4):
        yield np.array([i, i + 1])


def consume(a):
    pass


ds = tf.data.Dataset.from_generator(gen, tf.int32, (2,))
ds = ds.shuffle(4).prefetch(tf.data.experimental.AUTOTUNE)
for a in ds:
    assert a.dtype == tf.int32
    assert a.shape == (2,)
    consume(a)
