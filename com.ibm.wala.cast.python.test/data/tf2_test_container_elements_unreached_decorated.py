# Test `ContainerElements` where a function the call graph does not reach resizes a parameter of a function a decorator names (wala/ML#1255).
# The decorator may call it, so no list reports a length.
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 2


def register(f):
    return f


@register
def decorated(xs):
    xs.pop()


sink_list([x, y])
