# Test https://github.com/wala/ML/issues/1044.
import tensorflow as tf


def calls_replaced(m, t):
    return m.predict(t) + tf.reduce_sum(t)


def drop(o):
    d = delattr
    d(o, "predict")


inp = tf.keras.Input(shape=(3,))
out = tf.keras.layers.Dense(2)(inp)
xs = tf.ones((4, 3))

# Assigned over the method before any other use, but the attribute is deleted later through a
# local alias of `delattr`, which exposes the method again.
other = tf.keras.Model(inp, out)
other.predict = len
replaced = calls_replaced(other, xs)
assert float(replaced) == 16.0
drop(other)
again = calls_replaced(other, xs)
assert again.shape == (4, 2)
