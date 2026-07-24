# Companion to driver.py (wala/ML#370): the sidecar's claim about `y` disagrees with what the
# analysis infers, so the annotation is reported and NOT applied; inference wins.

import tensorflow as tf


def consume(y):
    pass


y = tf.ones((2, 2))
assert y.shape == (2, 2)
assert y.dtype == tf.float32
consume(y)
