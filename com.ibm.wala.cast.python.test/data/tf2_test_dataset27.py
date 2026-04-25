# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/data/Dataset#from_tensors.

import tensorflow as tf


def f(a):
    assert isinstance(a, tf.Tensor)


def g(a):
    assert isinstance(a, tf.Tensor)


def h(a):
    assert isinstance(a, tf.Tensor)


def i(a):
    assert isinstance(a, tf.Tensor)


dataset = tf.data.Dataset.from_tensors([1, 2, 3])

for element in dataset:
    assert isinstance(element, tf.Tensor)
    assert element.dtype == tf.int32
    assert element.shape == (3,)
    f(element)

    arg1 = element[0]
    assert isinstance(arg1, tf.Tensor)
    assert arg1.dtype == tf.int32
    assert arg1.shape == ()
    g(arg1)

    arg2 = element[1]
    assert isinstance(arg2, tf.Tensor)
    assert arg2.dtype == tf.int32
    assert arg2.shape == ()
    h(arg2)

    arg3 = element[2]
    assert isinstance(arg3, tf.Tensor)
    assert arg3.dtype == tf.int32
    assert arg3.shape == ()
    i(arg3)
