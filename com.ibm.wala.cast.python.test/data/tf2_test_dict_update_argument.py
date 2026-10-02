# Test https://github.com/wala/ML/issues/997: a dictionary updated from another dictionary the
# caller passed in gains that dictionary's fields, so a later read of one by its key sees the value.
import tensorflow as tf


def consume(x):
    assert x.shape == (3, 4) and x.dtype == tf.float32
    return x


def merge(target, extra):
    target.update(extra)
    consume(target["weight"])


merge({"name": "w"}, {"weight": tf.ones((3, 4))})
