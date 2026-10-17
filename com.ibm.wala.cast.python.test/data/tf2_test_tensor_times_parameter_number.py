# Test for wala/ML#1083: a tensor times a Python number computed from a parameter that holds Python
# numbers is typed as the tensor operand, inside the function as at module level.
import tensorflow as tf


def memory(j):
    return tf.range(8, dtype=tf.float32) * (j + 1.0)


def consume_loop(y):
    assert y.shape == (8,) and y.dtype == tf.float32
    return y


def consume_memory(y):
    assert y.shape == (8,) and y.dtype == tf.float32
    return y


for j in range(2):
    consume_loop(tf.range(8, dtype=tf.float32) * (j + 1.0))
    consume_memory(memory(j))
