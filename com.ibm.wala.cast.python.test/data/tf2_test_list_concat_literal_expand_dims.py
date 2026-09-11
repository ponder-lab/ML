"""Companion to `tf2_test_list_concat_expand_dims.py` (wala/ML#907): both operands of the list
concatenation are scalar literals, so the concatenation's length is the sum of theirs and
`tf.expand_dims` resolves to `(1, 3)`.
"""

import tensorflow as tf


def f(a):
    pass


def g(bos=3):
    prev = tf.expand_dims(([bos] + [1, 2]), 0)
    assert isinstance(prev, tf.Tensor)
    assert prev.shape == (1, 3)
    f(prev)


g()
