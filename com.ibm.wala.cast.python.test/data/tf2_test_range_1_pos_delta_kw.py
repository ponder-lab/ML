import tensorflow as tf


def f(a):
    pass


# 1 pos arg, delta as keyword
# pos 0 -> limit (because limit kw is missing)
# limit=1, delta=2, start=0
t = tf.range(1, delta=2)
assert isinstance(t, tf.Tensor)
assert t.shape == (1,)  # [0]
assert t.dtype == tf.int32

f(t)
