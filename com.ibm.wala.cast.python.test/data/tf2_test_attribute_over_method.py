# Test https://github.com/wala/ML/issues/1044.
import tensorflow as tf


def calls_replaced(m, t):
    return m.predict(t) + tf.reduce_sum(t)


def calls_kept(m, t):
    return m.predict(t) + tf.reduce_sum(t)


inp = tf.keras.Input(shape=(3,))
out = tf.keras.layers.Dense(2)(inp)
model = tf.keras.Model(inp, out)
model.compile(optimizer="sgd", loss="mse")

xs = tf.ones((4, 3))
ys = tf.ones((4, 2))
model.fit(xs, ys, epochs=1, verbose=0)

# Assigned over the method before any other use: every read sees `len`.
other = tf.keras.Model(inp, out)
other.predict = len
replaced = calls_replaced(other, xs)
assert float(replaced) == 16.0

# Assigned only after a use: that use reads the method.
late = tf.keras.Model(inp, out)
kept = calls_kept(late, xs)
assert kept.shape == (4, 2)
late.predict = len
