import tensorflow as tf


def consume_conv1d(t):
    pass


def consume_convolution1d(t):
    pass


def consume_conv2d(t):
    pass


def consume_convolution2d(t):
    pass


def consume_conv2d_transpose(t):
    pass


def consume_convolution2d_transpose(t):
    pass


def consume_convolution2d_field(t):
    pass


class Net(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv = tf.keras.layers.Convolution2D(4, 1)

    def call(self, x):
        y = self.conv(x)
        assert y.shape == (1, 8, 8, 4) and y.dtype == tf.float32, y
        consume_convolution2d_field(y)
        return y


assert tf.keras.layers.Convolution1D is tf.keras.layers.Conv1D
assert tf.keras.layers.Convolution2D is tf.keras.layers.Conv2D
assert tf.keras.layers.Convolution2DTranspose is tf.keras.layers.Conv2DTranspose

a = tf.keras.layers.Conv1D(4, 1)(tf.ones([1, 8, 3]))
assert a.shape == (1, 8, 4) and a.dtype == tf.float32, a
consume_conv1d(a)

b = tf.keras.layers.Convolution1D(4, 1)(tf.ones([1, 8, 3]))
assert b.shape == (1, 8, 4) and b.dtype == tf.float32, b
consume_convolution1d(b)

c = tf.keras.layers.Conv2D(4, 1)(tf.ones([1, 8, 8, 3]))
assert c.shape == (1, 8, 8, 4) and c.dtype == tf.float32, c
consume_conv2d(c)

d = tf.keras.layers.Convolution2D(4, 1)(tf.ones([1, 8, 8, 3]))
assert d.shape == (1, 8, 8, 4) and d.dtype == tf.float32, d
consume_convolution2d(d)

e = tf.keras.layers.Conv2DTranspose(4, 1)(tf.ones([1, 8, 8, 3]))
assert e.shape == (1, 8, 8, 4) and e.dtype == tf.float32, e
consume_conv2d_transpose(e)

f = tf.keras.layers.Convolution2DTranspose(4, 1)(tf.ones([1, 8, 8, 3]))
assert f.shape == (1, 8, 8, 4) and f.dtype == tf.float32, f
consume_convolution2d_transpose(f)

Net()(tf.ones([1, 8, 8, 3]))
