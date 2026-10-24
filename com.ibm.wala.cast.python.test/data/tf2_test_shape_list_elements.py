# The elements of a static shape's list are Python ints (wala/ML#1104), so a list built from them,
# as a `shape_list` utility builds one, may hold Python ints beside the tensor elements it takes
# from `tf.shape` for an axis of unknown size.
import tensorflow as tf


def consume_as_list(sizes):
    assert sizes == [2, 5, 8]


def consume_shape_list(sizes):
    assert sizes == [2, 5, 8]
    assert all(isinstance(s, int) for s in sizes)


def consume_tensor_list(sizes):
    assert all(isinstance(s, tf.Tensor) for s in sizes)


def shape_list(x):
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


x = tf.ones([2, 5, 8])
consume_as_list(x.shape.as_list())
consume_shape_list(shape_list(x))
# A control: a list of `tf.shape` elements alone holds no Python int.
consume_tensor_list([tf.shape(x)[i] for i in range(3)])
