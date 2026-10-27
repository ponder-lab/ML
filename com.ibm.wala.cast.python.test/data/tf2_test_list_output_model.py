# Calling a functional model whose outputs are a list gives each element of the result an unknown
# shape (wala/ML#1157).
import tensorflow as tf


def decode(conv_output):
    s = tf.shape(conv_output)
    return tf.reshape(conv_output, (s[0], s[1], s[1], 3, 7))


def consume(x):
    pass


def consume_first(x):
    pass


def consume_last(x):
    pass


def consume_each(x):
    pass


inp = tf.keras.layers.Input([26, 26, 3])
conv = tf.keras.layers.Conv2D(21, 1, strides=2)(inp)
model = tf.keras.Model(inp, [conv, decode(conv)])
res = model(tf.ones([4, 26, 26, 3]))
assert res[1].shape == (4, 13, 13, 3, 7)
assert res[1].dtype == tf.float32
consume(res[1])
assert res[0].shape == (4, 13, 13, 21)
consume_first(res[0])
consume_last(res[-1])
for output in res:
    consume_each(output)
