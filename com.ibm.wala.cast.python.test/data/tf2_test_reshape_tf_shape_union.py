import os

import tensorflow as tf


def consume(t):
    pass


# Fold-granularity witness (wala/ML#748): `f`'s parameter carries a two-member
# shape union — the concrete (2, 3, 8) constant and the dynamic-batch Keras
# input — through the conditional's φ, so the `tf.shape(x)` extractions fold
# against a multi-member source. Each member must keep its own axis
# classification (the concrete-sourced member folds to the constants, the
# dynamic-sourced one to *dynamic* on the batch axis), rather than the
# ambiguity joining every member to the wider marker.
def f(x):
    b = tf.shape(x)[0]
    s = tf.shape(x)[1]
    y = tf.reshape(x, [b, s, 8])
    consume(y)
    return y


flag = os.environ.get("ARIADNE_TEST_FLAG") == "1"

a = tf.ones((2, 3, 8))
inp = tf.keras.Input(shape=(3, 8))
x = inp if flag else a

out = f(x)

assert out.shape == (2, 3, 8)
assert out.dtype == tf.float32
