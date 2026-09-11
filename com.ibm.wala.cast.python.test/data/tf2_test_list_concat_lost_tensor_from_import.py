"""The from-import form of `tf2_test_list_concat_lost_tensor_expand_dims.py` (wala/ML#911): the unmodeled
API is bound to a bare name, so the call has no receiver read; the runtime value is still a broadcast
add of shape `(2, 3)` and `tf.expand_dims` yields `(1, 2, 3)`.
"""

import tensorflow as tf
from tensorflow import ensure_shape


def f(a):
    pass


def g():
    lost = ensure_shape(tf.ones((2, 3)), (2, 3))
    y = tf.expand_dims([1] + lost, 0)
    assert isinstance(y, tf.Tensor)
    assert y.shape == (1, 2, 3)
    assert y.dtype == tf.float32
    f(y)


g()
