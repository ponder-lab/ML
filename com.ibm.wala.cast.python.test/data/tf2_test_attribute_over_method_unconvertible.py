# Test https://github.com/wala/ML/issues/1044 beside a function whose SSA conversion fails.
import tensorflow as tf


def pad_in_time(x, padding_length):
    return tf.pad(x, [[0, 0], [0, padding_length], [0, 0]])


def align_in_time(x, length):
    time_dim = tf.shape(x)[1]
    return tf.cond(
        tf.less(time_dim, length),
        true_fn=lambda: pad_in_time(x, length - time_dim),
        false_fn=lambda: x[:, :length],
    )


def calls_replaced(m, t):
    return m.predict(t) + tf.reduce_sum(t)


inp = tf.keras.Input(shape=(3,))
out = tf.keras.layers.Dense(2)(inp)

# Assigned over the method before any other use: every read sees `len`.
other = tf.keras.Model(inp, out)
other.predict = len
replaced = calls_replaced(other, tf.ones((4, 3)))
assert float(replaced) == 16.0
