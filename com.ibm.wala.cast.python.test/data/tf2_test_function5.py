import tensorflow as tf


def func(t):
    pass


a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
assert a.shape == (2, 2)

func(a)
