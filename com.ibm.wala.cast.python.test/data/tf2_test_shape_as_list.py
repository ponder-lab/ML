import tensorflow as tf


def f(a):
    pass


# Shape-vector provenance (wala/ML#703): a tensor's full shape read into a
# Python list via `.shape.as_list()` (no slice) and consumed as another op's
# shape argument. The reshape's output shape is the source tensor's shape.
t = tf.ones((4, 5, 6))
x = tf.ones((120,))
r = tf.reshape(x, t.shape.as_list())
assert isinstance(r, tf.Tensor)
assert r.shape == (4, 5, 6)
assert r.dtype == tf.float32
f(r)
