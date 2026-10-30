# Test https://github.com/wala/ML/issues/1177: a project directory named `tensorflow` without an
# `__init__.py` does not shadow the library, which Python's path finder takes over the directory.
import tensorflow as tf
from tensorflow import ones


def consume(t):
    assert isinstance(t, tf.Tensor)
    assert t.shape == (1, 2)


def consume_attribute(t):
    assert isinstance(t, tf.Tensor)
    assert t.shape == (1, 2)


consume(ones([1, 2]))
consume_attribute(tf.ones([1, 2]))
