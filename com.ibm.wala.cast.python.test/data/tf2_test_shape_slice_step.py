import tensorflow as tf


def f(a):
    pass


def get_shape(t):
    return t.shape.as_list()


# Guard fixture (wala/ML#703): a non-unit step over a shape list is unmodeled,
# so the analysis soundly reports an unknown (⊤) shape. The asserts document
# the Python runtime truth ((4, 6), every other dim), not the analysis result.
t = tf.ones((4, 5, 6))
x = tf.ones((24,))
r = tf.reshape(x, get_shape(t)[::2])
assert r.shape == (4, 6)
assert r.dtype == tf.float32
f(r)
