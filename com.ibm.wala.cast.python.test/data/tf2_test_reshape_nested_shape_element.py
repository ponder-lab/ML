# Test https://github.com/wala/ML/issues/1062.
import tensorflow as tf


def size(flag):
    # A list concatenation on one arm and an int on the other: the dimension's points-to set holds
    # both, though only the int reaches the reshape at run time.
    return [1, 2] + [3] if flag else 4


def consume(t):
    return t


def project(inputs, flag):
    outputs = tf.reshape(inputs, [size(flag), -1])
    return consume(outputs)


result = project(tf.ones((2, 3, 4)), False)
assert result.shape == (4, 6)
assert result.dtype == tf.float32
