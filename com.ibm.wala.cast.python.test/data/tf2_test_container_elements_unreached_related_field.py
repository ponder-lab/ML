# Test `ContainerElements` where a function the call graph does not reach resizes an attribute, `self.items`, under whose name a subclass stores a tensor (wala/ML#1268).
# `self` may be a `Sub`, whose tensor may have a method `append` that changes a list it is passed, so no list reports a length.
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 2


class Holder:
    def build(self):
        self.items.append(1)


class Sub(Holder):
    def __init__(self):
        self.items = tf.ones([4, 2])


s = Sub()
sink_list([x, y])
