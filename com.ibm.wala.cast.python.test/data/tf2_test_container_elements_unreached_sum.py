# Test `ContainerElements` where a function the call graph does not reach resizes a sum of a list it builds, `list(a) + [1]` (wala/ML#1255).
# The sum is a new list, so a list keeps its length.
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 2


def never_called(a):
    joined = list(a) + [1]
    joined.append(1)


sink_list([x, y])
