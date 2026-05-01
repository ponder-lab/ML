import tensorflow as tf


def add(a, b):
    return a + b


dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])

for element in dataset:
    assert element.shape == ()
    assert element.dtype == tf.int32
    c = add(element, element)
