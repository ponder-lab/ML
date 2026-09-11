"""The ndarray-method form of `tf2_test_list_concat_lost_tensor_expand_dims.py` (wala/ML#911): an
unmodeled method on an array allocated under the `numpy` namespace. `cumsum` flattens to `(6,)`, the
concatenation is a broadcast add of shape `(6,)`, and `tf.expand_dims` yields `(1, 6)`.
"""

import numpy as np
import tensorflow as tf


def f(a):
    pass


def g():
    lost = np.ones((2, 3)).cumsum()
    y = tf.expand_dims([1] + lost, 0)
    assert isinstance(y, tf.Tensor)
    assert y.shape == (1, 6)
    f(y)


g()
