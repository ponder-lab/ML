import tensorflow as tf


def f(x):
    # Eager, no decorator: the parameter is exactly the argument.
    assert x.dtype == tf.int32
    assert x.shape.as_list() == [3]
    return x


f(tf.constant([1, 2, 3], dtype=tf.int32))
