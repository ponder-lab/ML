# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/data/Dataset#filter.

import tensorflow as tf


def f(a):
    assert isinstance(a, tf.Tensor)


def g(a):
    assert isinstance(a, tf.Tensor)


dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
dataset = dataset.filter(lambda x: x < 3)

for element in dataset:
    assert isinstance(element, tf.Tensor)
    assert element.dtype == tf.int32
    assert element.shape == ()
    f(element)


# `tf.math.equal(x, y)` is required for equality comparison
def filter_fn(x):
    return tf.math.equal(x, 1)


dataset = dataset.filter(filter_fn)

for element in dataset:
    assert isinstance(element, tf.Tensor)
    assert element.dtype == tf.int32
    assert element.shape == ()
    g(element)
