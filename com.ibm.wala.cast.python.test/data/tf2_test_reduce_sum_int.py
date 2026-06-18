import tensorflow as tf

# Guards that `tf.reduce_sum` preserves an integer input's dtype (no float32 promotion). Unlike
# `tf.reduce_mean`, summing integers yields an integer (wala/ML#514).


def f(x):
    pass


a = tf.constant([1, 2, 3], dtype=tf.int32)
r = tf.reduce_sum(a)
assert r.dtype == tf.int32
assert r.shape == ()
f(r)
