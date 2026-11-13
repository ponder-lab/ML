# Test `ContainerElements` where a function the call graph does not reach resizes a slice of an object whose class defines `__getitem__` (wala/ML#1255).
# The slice may be any list, so no list reports a length.
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 2


class Shared:
    def __getitem__(self, key):
        return []


def never_called():
    copied = Shared()[:]
    copied.append(1)


sink_list([x, y])
