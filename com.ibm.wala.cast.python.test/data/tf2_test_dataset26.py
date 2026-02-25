# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/data/Dataset#enumerate.

import tensorflow as tf


def f(a):
    assert isinstance(a, tf.Tensor)


def g1(a):
    assert isinstance(a, tf.Tensor)


def g2(a):
    assert isinstance(a, tf.Tensor)


def g3(a):
    assert isinstance(a, tf.Tensor)


def h(a):
    assert isinstance(a, tuple)


dataset = tf.data.Dataset.from_tensor_slices([(7, 8), (9, 10)])
dataset = dataset.enumerate()

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
    assert arg2.shape == (2,)
    g1(arg2)

    assert len(element[1] == 2)
    arg3 = element[1][0]
    assert isinstance(arg3, tf.Tensor)
    assert arg3.dtype == tf.int32
    assert arg3.shape == ()
    g2(arg3)

    arg4 = element[1][1]
    assert isinstance(arg4, tf.Tensor)
    assert arg4.dtype == tf.int32
    assert arg4.shape == ()
    g3(arg4)

    h(element)
