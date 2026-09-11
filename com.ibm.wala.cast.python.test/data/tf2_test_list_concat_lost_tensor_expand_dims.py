"""The uncovered caller of the wala/ML#907 sequence-concatenation stage: `[1] + lost` where `lost` is
a tensor the analysis has no evidence for (here the result of an unmodeled API, `tf.ensure_shape`).

At runtime Python tries the tensor's `__radd__` before list concatenation, so the value is a
broadcast add of shape `(2, 3)` and `tf.expand_dims` yields `(1, 2, 3)`. The analysis sees no
tensor evidence on either operand, so the concatenation stage types the value as a rank-1
sequence instead.
"""

import tensorflow as tf


def f(a):
    pass


def g():
    lost = tf.ensure_shape(tf.ones((2, 3)), (2, 3))
    y = tf.expand_dims([1] + lost, 0)
    assert isinstance(y, tf.Tensor)
    assert y.shape == (1, 2, 3)
    assert y.dtype == tf.float32
    f(y)


g()
