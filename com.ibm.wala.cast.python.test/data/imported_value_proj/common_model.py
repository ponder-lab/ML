import tensorflow as tf


def consume(t):
    assert t.shape == (2, 4)
    assert t.dtype == tf.float32
    return t


class Inner(tf.keras.layers.Layer):

    def call(self, x):
        return consume(x)


class Net(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.inner = Inner()

    def call(self, x):
        return self.inner(x)


model = Net()


def get_model():
    return model


class Plain:

    def m(self, t):
        return consume(t)


plain = Plain()

ones = tf.ones([2, 4])

scaled = tf.ones([2, 4])
scaled *= 2

batch = 2
