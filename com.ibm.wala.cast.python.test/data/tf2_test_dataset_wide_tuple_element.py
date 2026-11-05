# Test the members of a dataset's tuple element of five components, unpacked in a loop: each keeps
# its own type, the fifth too (wala/ML#1221).
import random

import numpy as np
import tensorflow as tf


def consume_sliced_fifth(e):
    pass


def consume_generated_fifth(e):
    pass


sliced = tf.data.Dataset.from_tensor_slices(
    (
        tf.ones([8, 1]),
        tf.ones([8, 2]),
        tf.ones([8, 3]),
        tf.ones([8, 4]),
        tf.zeros([8, 5], tf.int32),
    )
)
for a, b, c, d, e in sliced:
    assert e.shape == (5,) and e.dtype == tf.int32
    consume_sliced_fifth(e)


def five():
    for _ in range(4):
        yield (
            np.int32(0),
            np.int32(1),
            np.int32(2),
            np.int32(3),
            np.arange(random.randint(1, 4), dtype=np.int32),
        )


generated = tf.data.Dataset.from_generator(
    five, (tf.int32, tf.int32, tf.int32, tf.int32, tf.int32)
)
for a, b, c, d, e in generated:
    assert e.dtype == tf.int32 and e.shape.rank == 1
    consume_generated_fifth(e)
