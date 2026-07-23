# Probe for wala/ML#746: a string-guarded dispatch whose arms return different ranks. Each
# call site's mode constant (the keyword at the second site, the materialized default at the
# first) decides its arm, so neither sink sees the other site's member.
import tensorflow as tf


def consume(t):
    pass


def consume2(t):
    pass


def dispatch(x, mode="embedding"):
    if mode == "embedding":
        return tf.expand_dims(x, -1)
    elif mode == "projection":
        return tf.reshape(x, (6,))
    else:
        raise ValueError("mode {} is not valid.".format(mode))


a = dispatch(tf.ones((2, 3)))
b = dispatch(tf.ones((2, 3)), mode="projection")
consume(a)
consume2(b)

assert a.shape == (2, 3, 1)
assert b.shape == (6,)
assert a.dtype == tf.float32
assert b.dtype == tf.float32
