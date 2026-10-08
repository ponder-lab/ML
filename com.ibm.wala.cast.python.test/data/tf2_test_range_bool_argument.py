# Test `tf.range` over an argument whose possible values include a Python bool: `tf.range` rejects a
# bool limit at run time, so that value yields no tensor and the integer one still gives the shape.
# A `False` sentinel for "no length" that reaches the range call along an untaken path has this
# shape.
import tensorflow as tf


def consume(x):
    assert x.shape == (4,) and x.dtype == tf.int32
    return x


def make(n):
    return tf.range(n)


def maybe(flag):
    n = 4 if flag else False
    if n is not False:
        consume(make(n))


maybe(True)
maybe(False)
