# Test `ContainerElements` where a function the call graph does not reach adds to a parameter in place, `a += b`, then resizes the result (wala/ML#1255).
# The result is the parameter, which may be any list, so no list reports a length.
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 2


def extend_and_grow(a, b):
    a += b
    a.append(1)


handlers = [extend_and_grow]

sink_list([x, y])
