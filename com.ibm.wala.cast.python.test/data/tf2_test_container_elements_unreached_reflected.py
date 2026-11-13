# Test `ContainerElements` where a function the call graph does not reach resizes a sum whose right operand defines `__radd__` (wala/ML#1255).
# The reflected operator may return any list, so no list reports a length.
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 2


class Shared:
    def __radd__(self, other):
        return other


def never_called():
    joined = [0] + Shared()
    joined.append(1)


sink_list([x, y])
