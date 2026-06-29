import tensorflow as tf
import numpy as np

signature = [tf.TensorSpec(shape=(3,), dtype=tf.int32)]


def consume(x):
    # The runtime truth: the input_signature pins the parameter to int32. Ariadne
    # should infer the same, but currently lands at unknown (wala/ML#629).
    assert x.shape == (3,)
    assert x.dtype == tf.int32


@tf.function(input_signature=signature)
def f(x):
    consume(x)
    return x


# The argument is a tensor with a known shape (3,) but a dtype Ariadne cannot
# resolve (numpy.array with no dtype= lands at unknown). The int32 input_signature
# pins the parameter's dtype, but Ariadne does not consume input_signature, so the
# parameter's dtype stays unknown rather than int32. wala/ML#629.
f(np.array([1, 2, 3]))
