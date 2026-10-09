# Test `tf.abs` of a complex tensor: the result is real, `float32` for `complex64`, while a real
# input keeps its dtype.
import tensorflow as tf


def consume_complex(t):
    assert t.dtype == tf.float32
    assert t.shape == (3,)


def consume_real(t):
    assert t.dtype == tf.float32
    assert t.shape == (2,)


x = tf.cast(tf.constant([1.0, 2.0, 3.0]), tf.complex64)
consume_complex(tf.abs(x))
consume_real(tf.abs(tf.constant([1.0, -2.0])))
