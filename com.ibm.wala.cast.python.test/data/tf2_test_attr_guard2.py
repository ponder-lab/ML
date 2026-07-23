# Probe for wala/ML#762 driving wala/ML#761: the guard attribute takes its value from an
# unpassed constructor default, so the arm decision requires the default to bind at the
# class-instantiation call.
import tensorflow as tf


class Dispatcher:
    def __init__(self, use_expand=True):
        self.use_expand = use_expand

    def run(self, x):
        if self.use_expand:
            return tf.expand_dims(x, -1)
        else:
            return tf.reshape(x, (6,))


def consume(t):
    pass


d = Dispatcher()
y = d.run(tf.ones((2, 3)))
consume(y)

assert y.shape == (2, 3, 1)
assert y.dtype == tf.float32
