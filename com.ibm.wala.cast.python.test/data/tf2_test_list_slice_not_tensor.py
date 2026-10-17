# Test for wala/ML#1121: a slice of a Python list of tensors is a list, not a tensor, while a slice
# of a tensor is a tensor.
import tensorflow as tf


def consume_sub(sub):
    assert isinstance(sub, list) and len(sub) == 2
    return sub


def consume_tuple_sub(sub):
    assert isinstance(sub, tuple) and len(sub) == 2
    return sub


def consume_tensor_slice(t):
    assert t.dtype == tf.float32 and t.shape == (2, 4)
    return t


def split_last(inputs):
    return consume_sub(inputs[0:-1])


def split_last_tuple(inputs):
    return consume_tuple_sub(inputs[0:-1])


split_last([tf.ones([2]), tf.ones([3]), tf.ones([4])])
split_last_tuple((tf.ones([2]), tf.ones([3]), tf.ones([4])))
consume_tensor_slice(tf.ones([3, 4])[0:-1])
