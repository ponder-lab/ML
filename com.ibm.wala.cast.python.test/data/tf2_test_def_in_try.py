# Test https://github.com/wala/ML/issues/1004: a function defined inside a `try` body, whose own
# body makes a call, does not stop its module from translating.
import tensorflow as tf


def sink(x):
    assert x.shape == (2,)
    return x


try:

    def inner(y):
        return tf.identity(y)

    sink(inner(tf.ones(2)))
except:
    pass
