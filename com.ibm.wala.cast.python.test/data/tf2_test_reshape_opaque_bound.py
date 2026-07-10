import tensorflow as tf


def consume(t):
    pass


def get_shape_list(tensor):
    return tensor.shape.as_list()


# A reshape whose target is a shape-vector slice with an opaque bound
# (wala/ML#711): the walk cannot resolve `k`, so the output shape is unknown,
# but the input's dtype must survive.
def f(t, k):
    return tf.reshape(t, get_shape_list(t)[0:k])


x = tf.ones((2, 4, 6))
out = f(x, x.shape.ndims)

assert out.shape == (2, 4, 6)
assert out.dtype == tf.float32

consume(out)
