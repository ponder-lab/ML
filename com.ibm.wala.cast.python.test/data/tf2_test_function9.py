import tensorflow as tf


def func(t):
    pass


a = tf.constant([[1.0, 3.0]])
assert a.shape == (1, 2)

func(a)
