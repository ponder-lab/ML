import tensorflow as tf
import numpy as np


@tf.function(input_signature=[tf.TensorSpec(shape=(3,), dtype=tf.int32)])
def f(x):
    # The numpy argument is int64, but the signature coerces the parameter to
    # int32 (traced). So the parameter's dtype comes from the signature.
    assert x.dtype == tf.int32
    assert x.shape.as_list() == [3]
    return x


f(np.array([1, 2, 3]))
