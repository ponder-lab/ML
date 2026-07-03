# Miniature of NLPGNN's `tests/TG/EN/generation.py` (wala/ML#678): a nested entry script defining
# `GenGPT2` (same-named across siblings) over the root-level `nlpgnn` package.
import tensorflow as tf

from nlpgnn.models import gpt2
from nlpgnn.sample import samples


class GenGPT2(tf.keras.Model):
    def __init__(self, param, **kwargs):
        super(GenGPT2, self).__init__(**kwargs)
        self.model = gpt2.GPT2(param)

    def call(self, inputs, past=None, is_training=True):
        out = self.model(inputs, past, is_training)
        return out

    def predict(self, inputs, past=None, is_training=False):
        return self(inputs, past, is_training)


def consume(t):
    pass


model = GenGPT2("generation")
x = tf.ones((2, 2))
out = samples.sample_sequence(model, x, length=40)
consume(out)

assert out.shape == (2, 8)
assert out.dtype == tf.float32
