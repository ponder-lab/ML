# Probe for wala/ML#618: `tf.reshape` with (a) a literal shape list and (b) a shape list built at
# runtime by list concatenation (`[tf.shape(x)[0], tf.shape(x)[1]] + [n]`, the vendored `Conv1d`
# idiom).
import tensorflow as tf


def consume(t):
    pass


def consume2(t):
    pass


x = tf.ones((2, 3, 8))

flat = tf.reshape(x, [-1, 8])
consume(flat)
assert flat.shape == (6, 8)

output_shape = [tf.shape(x)[0], tf.shape(x)[1]] + [16]
y = tf.reshape(tf.ones((2, 3, 16)), output_shape)
consume2(y)
assert y.shape == (2, 3, 16)
