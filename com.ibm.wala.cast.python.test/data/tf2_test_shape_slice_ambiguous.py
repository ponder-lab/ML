import tensorflow as tf


def f(a):
    pass


def get_shape(t):
    return t.shape.as_list()


def tail(t, flag):
    k = 1 if flag else 2
    return tf.reshape(tf.ones((30,)), get_shape(t)[k:])


# Guard fixture (wala/ML#703): the slice bound `k` is a φ of two constants, so
# within one context its points-to set is ambiguous and the analysis must not
# assert either slicing; it soundly reports an unknown (⊤) shape. The asserts
# document the Python runtime truth for the taken branch.
t = tf.ones((4, 1, 30))
a = tail(t, True)
assert a.shape == (1, 30)
assert a.dtype == tf.float32
f(a)
