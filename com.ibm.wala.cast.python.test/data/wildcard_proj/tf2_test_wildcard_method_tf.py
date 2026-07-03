# Probe for wala/ML#684: this script never imports tensorflow directly; `tf` arrives through
# `from helpers import *` (a wildcard re-export of an import binding) and is read inside a
# name-mangled @staticmethod of a keras.Model subclass, called self-qualified — the subject's
# `MusicTransformer.__prepare_train_data` shape.
from helpers import *


class M(tf.keras.Model):
    def __init__(self, **kwargs):
        super(M, self).__init__(**kwargs)

    def call(self, inputs, is_training=True):
        return inputs

    @staticmethod
    def __prepare_train_data(y):
        start_token = tf.ones((2, 1), dtype=y.dtype) * 3
        return start_token

    def prep(self, y):
        return self.__prepare_train_data(y)


def consume(t):
    pass


m = M()
y = tf.ones((2, 4), dtype=tf.int32)
out = m.prep(y)
consume(out)

assert out.shape == (2, 1)
assert out.dtype == tf.int32
