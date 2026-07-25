# The corpus loader chain shape (wala/ML#776): `from_generator` with declared output types,
# then `.shuffle(...).prefetch(...).window(n)`. `window` yields datasets of datasets; the
# element dtype declaration must survive the combinator chain to reach the iterated values.
import numpy as np
import tensorflow as tf


def gen():
    for i in range(4):
        yield np.array([i, i + 1])


def consume(w):
    pass


ds = (
    tf.data.Dataset.from_generator(gen, tf.int32)
    .shuffle(4)
    .prefetch(tf.data.experimental.AUTOTUNE)
    .window(2)
)
for w in ds:
    for e in w:
        assert e.dtype == tf.int32
        consume(e)
