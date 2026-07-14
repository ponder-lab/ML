import tensorflow as tf


def f(a):
    pass


def get_shape(t):
    return t.shape.as_list()


# Negative-step companion of tf2_test_shape_slice_step.py (wala/ML#709): constant
# negative steps reverse and stride the shape list under Python's adjusted-indices
# semantics, covering absent, explicit, negative, and out-of-range bounds; the
# out-of-range positive-step case pins the clamping arms on that side too.
t = tf.ones((4, 5, 6))

r1 = tf.reshape(tf.ones((24,)), get_shape(t)[::-2])
assert r1.shape == (6, 4)
r2 = tf.reshape(tf.ones((30,)), get_shape(t)[2:0:-1])
assert r2.shape == (6, 5)
r3 = tf.reshape(tf.ones((20,)), get_shape(t)[1::-1])
assert r3.shape == (5, 4)
r4 = tf.reshape(tf.ones((120,)), get_shape(t)[-9:9])
assert r4.shape == (4, 5, 6)

f(r1)
f(r2)
f(r3)
f(r4)
