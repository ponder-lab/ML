import tensorflow as tf


def f(a):
    pass


# `tf.linspace` with integer start/stop promotes to float64 (TF 2.9). The
# float-input variant is covered by `tf2_test_linspace.py`; this fixture
# exercises the int-promotion branch in `Linspace.getDefaultDTypes`.
y = tf.linspace(tf.constant(0, dtype=tf.int32), tf.constant(10, dtype=tf.int32), 5)
assert isinstance(y, tf.Tensor)
assert y.shape == (5,)
assert y.dtype == tf.float64

f(y)
