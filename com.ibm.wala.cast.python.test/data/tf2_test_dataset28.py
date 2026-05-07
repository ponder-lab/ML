# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/data/Dataset#from_tensors.

import tensorflow as tf


def f(a):
    assert isinstance(a, tf.Tensor)


def g(a):
    assert isinstance(a, tf.Tensor)


def h(a):
    assert isinstance(a, tuple)


dataset = tf.data.Dataset.from_tensors(([1, 2, 3], "A"))

for element in dataset:
    arg1 = element[0]
    assert isinstance(arg1, tf.Tensor)
    assert arg1.dtype == tf.int32
    assert arg1.shape == (3,)
    f(arg1)

    arg2 = element[1]
    assert isinstance(arg2, tf.Tensor)
    assert arg2.dtype == tf.string
    assert arg2.shape == ()
    g(arg2)

    assert isinstance(element, tuple)
    assert len(element) == 2
    assert isinstance(element[0], tf.Tensor)
    assert isinstance(element[1], tf.Tensor)
    assert element[0].dtype == tf.int32
    assert element[0].shape == (3,)
    assert element[1].dtype == tf.string
    assert element[1].shape == ()
    h(element)
