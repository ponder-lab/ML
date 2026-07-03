import tensorflow as tf

from helpers.samples import sample_sequence


class GenGPT2(tf.keras.Model):
    def __init__(self, param=None, **kwargs):
        super(GenGPT2, self).__init__(**kwargs)
        self.param = param

    def call(self, inputs, is_training=True):
        return inputs + 1.0

    def predict(self, inputs):
        return self(inputs)


def consume(t):
    pass


model = GenGPT2("a")
x = tf.ones((2, 2))
out = sample_sequence(model, x)
consume(out)

assert out.shape == (2, 2)
assert out.dtype == tf.float32
