import tensorflow as tf


def f(a):
    pass


# Drives the `output_type=tf.int32` branch of `Argmax.getDTypes`. Default
# argmax output dtype is `int64`; passing `output_type=tf.int32` should
# override it to `int32`. The static analysis reads `output_type` from
# `tensorflow.xml`'s `paramNames` and routes through the inherited
# dtype-arg machinery.
x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.math.argmax(x, axis=0, output_type=tf.int32)
assert isinstance(y, tf.Tensor)
assert y.dtype == tf.int32

f(y)
