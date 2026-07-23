# Probe for wala/ML#761: a guard over a constant instance attribute decides its arm, so the
# untaken arm's differently-ranked member never reaches the sink.
import tensorflow as tf


class Dispatcher:
    def __init__(self):
        self.use_expand = True

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
