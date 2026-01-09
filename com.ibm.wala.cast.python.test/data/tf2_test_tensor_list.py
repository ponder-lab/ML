import tensorflow as tf


def add(a, b):
    return a + b


list = [tf.ones([1, 2]), tf.ones([2, 2])]

assert list[0].shape == (1, 2)
assert list[1].shape == (2, 2)

assert list[0].dtype == tf.float32
assert list[1].dtype == tf.float32

for element in list:
    c = add(element, element)
