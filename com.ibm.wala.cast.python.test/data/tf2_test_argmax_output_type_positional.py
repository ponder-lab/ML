import tensorflow as tf


def f(x, y):
    pass


# Counterpart of `tf2_test_argmax_output_type.py` that passes `output_type`
# POSITIONALLY (`tf.math.argmax(x, 0, tf.int32)`). Shape inference must not
# misread the positional `output_type` argument as `keepdims` (they share
# positional index 2 in `ReduceMean`); argmax always removes the scanned axis.
x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
assert x.shape == (2, 3)
assert x.dtype == tf.float32
y = tf.math.argmax(x, 0, tf.int32)
assert isinstance(y, tf.Tensor)
assert y.shape == (3,)
assert y.dtype == tf.int32

f(x, y)
