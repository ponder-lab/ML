# Test enumerate. The first element of the tuple returned isn't a tensor.

import tensorflow as tf


def f(a):
    pass


def g(a):
    pass


dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])

for step, element in enumerate(dataset, 1):
    assert isinstance(step, int)
    f(step)

    assert isinstance(element, tf.Tensor)
    assert element.shape == ()
    assert element.dtype == tf.int32
    g(element)
