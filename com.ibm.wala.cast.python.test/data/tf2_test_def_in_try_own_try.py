# Test https://github.com/wala/ML/issues/1004: a function defined inside a `try` body keeps the
# handlers of its own `try`.
import tensorflow as tf


def sink(x):
    assert x.shape == (2,)
    return x


def fallback(x):
    assert x.shape == (2,)
    return x


try:

    def guarded(y):
        try:
            return sink(tf.identity(y))
        except:
            return fallback(y)

    guarded(tf.ones(2))
except:
    pass
