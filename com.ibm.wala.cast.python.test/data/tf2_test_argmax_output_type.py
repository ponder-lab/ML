import tensorflow as tf


def f(x, y):
    pass


# Drives the `output_type=tf.int32` branch of `Argmax.getDTypes`. Default
# argmax output dtype is `int64`; passing `output_type=tf.int32` should
# override it to `int32`. The static analysis reads `output_type` from
# `tensorflow.xml`'s `paramNames` and routes through the inherited
# dtype-arg machinery.
x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
assert x.shape == (2, 3)
assert x.dtype == tf.float32
y = tf.math.argmax(x, axis=0, output_type=tf.int32)
assert isinstance(y, tf.Tensor)
assert y.shape == (3,)
assert y.dtype == tf.int32

f(x, y)
