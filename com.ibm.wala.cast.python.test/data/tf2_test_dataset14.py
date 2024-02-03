# Test enumerate. The first element of the tuple returned isn't a tensor.

import tensorflow as tf


def f(a):
    pass


def g(a):
    pass


def h(ds):
    for step, element in enumerate(ds, 1):
        f(step)
        g(element)


dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
h(dataset)
