# A `matmul` with a transposing flag, whose type feed reads the flag as its generator does
# (wala/ML#1164).
import tensorflow as tf


def consume(t):
    assert t.shape == (4, 5) and t.dtype == tf.float32
    return t


consume(tf.matmul(tf.ones([4, 3]), tf.ones([5, 3]), transpose_b=True))
