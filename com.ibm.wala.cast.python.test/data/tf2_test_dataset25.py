# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/data/Dataset#enumerate.

import tensorflow as tf


def f(a):
    assert isinstance(a, tf.Tensor)


def g(a):
    assert isinstance(a, tf.Tensor)


def h(a):
    assert isinstance(a, tuple)


dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
dataset = dataset.enumerate(start=5)

for element in dataset:
    assert isinstance(element, tuple)
    assert len(element) == 2

    arg1 = element[0]
    assert isinstance(arg1, tf.Tensor)
    assert arg1.dtype == tf.int64
    assert arg1.shape == ()
    f(arg1)

    arg2 = element[1]
    assert isinstance(arg2, tf.Tensor)
    assert arg2.dtype == tf.int32
    assert arg2.shape == ()
    g(arg2)

    h(element)
