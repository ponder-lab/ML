import tensorflow as tf


def check_result(res):
    pass


a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])
c = a / b
assert c.shape == (2,)
assert c.dtype == tf.float32
check_result(c)
