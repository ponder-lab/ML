# Test https://github.com/wala/ML/issues/1062.
import tensorflow as tf


def consume(t):
    return t


def project(inputs):
    # The constant's value is a list the concatenation model synthesizes, with no literal
    # dimension of its own.
    outputs = tf.reshape(inputs, tf.constant([4] + [6]))
    return consume(outputs)


result = project(tf.ones((2, 3, 4)))
assert result.shape == (4, 6)
assert result.dtype == tf.float32
