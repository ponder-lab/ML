# Test `ContainerElements` where a function the call graph does not reach resizes an attribute, `self.items`, under whose name a class whose second base is an alias of the method's class stores a tensor (wala/ML#1268).
# The analysis does not resolve `Alias`, so a `Both` may be an `Other`, its tensor may be the value `Other.build` appends to, and no list reports a length.
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 2


class Holder:
    def __init__(self):
        self.items = tf.ones([4, 2])


class Other:
    def build(self):
        self.items.append(1)


Alias = Other


class Both(Holder, Alias):
    pass


b = Both()
sink_list([x, y])
