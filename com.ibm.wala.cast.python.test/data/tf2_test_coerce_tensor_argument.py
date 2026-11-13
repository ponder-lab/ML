# Test https://github.com/wala/ML/issues/1253: a parameter fed a TensorFlow tensor keeps the
# tensor's dtype where an operation beside a tensor of another dtype would impose one, since
# TensorFlow converts only an argument that is no tensor, and such an operation raises when it runs.
import sys

import numpy as np
import tensorflow as tf

flag = len(sys.argv) > 99


def scale_tensor(x):
    if flag:
        # Never runs: a `float32` tensor times an `int32` one raises.
        return x * tf.constant([1], tf.int32)
    return x


def scale_array(x):
    # A NumPy argument is converted to the dtype of the tensor beside it.
    return x * tf.constant([1.0], tf.float32)


assert scale_tensor(tf.ones([2])).dtype == tf.float32
assert scale_array(np.ones([2])).dtype == tf.float32
