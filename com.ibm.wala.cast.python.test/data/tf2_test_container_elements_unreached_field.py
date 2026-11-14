# Test `ContainerElements` where a function the call graph does not reach resizes an attribute it stores a list it builds under, `self.items` (wala/ML#1255).
# No list the analysis has is stored under `items`, so a list keeps its length, beside a class
# whose `__setattr__` stores into its items, of which `Holder` is no subclass.
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 2


class Items(dict):
    __setattr__ = dict.__setitem__


class Holder:
    def build(self):
        self.items = []
        self.items.append(1)


sink_list([x, y])
