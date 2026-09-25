# Control for wala/ML#958: a parameter genuinely fed an int32 and a float32 elementwise result at
# one shape keeps both members; the operands' dataflow dtypes decide each add's dtype separately.
import tensorflow as tf


def take(arr):
    pass


a = tf.constant([[1, 2, 3], [4, 5, 6]])
b = tf.constant([[1, 1, 1], [1, 1, 1]])
ints = a + b
assert ints.shape == (2, 3) and ints.dtype == tf.int32
take(ints)

c = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
d = tf.constant([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
floats = c + d
assert floats.shape == (2, 3) and floats.dtype == tf.float32
take(floats)
