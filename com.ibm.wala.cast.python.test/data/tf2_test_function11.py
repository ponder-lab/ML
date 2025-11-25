import tensorflow as tf


def func(t):
    pass


a = tf.constant(
    [
        [[1, 2, 3], [5, 6, 7], [9, 10, 11]],
        [[13, 14, 15], [17, 18, 19], [21, 22, 23]],
    ]
)
assert a.shape == (2, 3, 3)
assert a.dtype == tf.int32

func(a)
