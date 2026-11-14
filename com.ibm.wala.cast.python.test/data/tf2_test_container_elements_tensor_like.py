# A list times a tensor or an array is the tensor's or the array's multiplication, which makes
# no list, while a list times an integer is a repetition, which makes one (wala/ML#960,
# wala/ML#1259).
import numpy as np
import tensorflow as tf


def sink_array_times_list(a):
    assert isinstance(a, np.ndarray)


def sink_tensor_times_list(t):
    assert isinstance(t, tf.Tensor)


def sink_list_times_int(c):
    assert isinstance(c, list) and len(c) == 4


sink_array_times_list(np.ones(2) * [2, 3])
sink_tensor_times_list(tf.constant([1.0, 2.0]) * [2.0, 3.0])
sink_list_times_int([2, 3] * 2)
