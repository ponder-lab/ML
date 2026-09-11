"""The submodule form of `tf2_test_list_concat_lost_tensor_expand_dims.py` (wala/ML#911): the unmodeled
API hangs off `tf.image`, which the summaries allocate as a plain object; the `tensorflow` module at the
root of the attribute chain is the only library-typed value. The runtime value is a broadcast add of
shape `(2, 3, 1)` and `tf.expand_dims` yields `(1, 2, 3, 1)`.
"""

import tensorflow as tf


def f(a):
    pass


def g():
    lost = tf.image.rgb_to_grayscale(tf.ones((2, 3, 3)))
    y = tf.expand_dims([1] + lost, 0)
    assert isinstance(y, tf.Tensor)
    assert y.shape == (1, 2, 3, 1)
    assert y.dtype == tf.float32
    f(y)


g()
