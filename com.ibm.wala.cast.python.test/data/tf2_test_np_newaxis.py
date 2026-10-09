# Test `np.newaxis` in a tensor subscript: it is `None` at runtime, as `tf.newaxis` is, and inserts a
# size-1 axis.
import numpy as np
import tensorflow as tf


def consume_inner(t):
    assert t.shape == (2, 1, 1, 1, 150, 4)


def f(bboxes):
    consume_inner(bboxes[:, np.newaxis, np.newaxis, np.newaxis, :, :])


f(tf.ones((2, 150, 4)))
