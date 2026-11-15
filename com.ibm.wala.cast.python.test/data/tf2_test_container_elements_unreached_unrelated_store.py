# Test `ContainerElements` where a function the call graph does not reach resizes an attribute, `self.items`, under whose name an unreached method of an unrelated class stores a value a call returns (wala/ML#1268).
# `Other.load` stores on an `Other`, which can't be the `Holder` `Holder.build` appends to, so a list keeps its length.
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 2


def make():
    return tf.ones([4, 2])


class Other:
    def load(self):
        self.items = make()


class Holder:
    def build(self):
        self.items = []
        self.items.append(1)


sink_list([x, y])
