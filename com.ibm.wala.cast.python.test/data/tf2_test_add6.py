import tensorflow as tf


def f(a):
    pass


x = [1, 2, 3, 4, 5]
y = tf.constant([1, 2, 3, 4, 5])
z = tf.math.add(x, y)
assert z.shape.as_list() == [5]
assert z.dtype == tf.int32
f(z)
