# Test `ContainerElements` where a function the call graph does not reach resizes a parameter, where nothing names the function (wala/ML#1255).
# Only a caller outside the program can call it, so a list keeps its length.
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 2


def never_named(xs):
    xs.pop()


sink_list([x, y])
