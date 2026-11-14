# Test `ContainerElements` where a function the call graph does not reach resizes a parameter of a function a string names (wala/ML#1255).
# A `getattr` by that string may call it, so no list reports a length.
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 2


import sys


def named_by_string(xs):
    xs.pop()


lookup = getattr(sys.modules[__name__], "named_by_string")

sink_list([x, y])
