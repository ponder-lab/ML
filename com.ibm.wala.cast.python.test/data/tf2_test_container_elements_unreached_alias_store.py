# Test `ContainerElements` where a function the call graph does not reach resizes an attribute, `self.items`, under whose name an unreached method of another class stores a value a call returns, and a class has both classes as bases, one of them through an alias (wala/ML#1268).
# The analysis does not resolve `Alias`, so a `Both` may be a `Holder` and an `Other`, the value `Holder.load` stores may be the one `Other.build` appends to, and no list reports a length.
import tensorflow as tf

x = tf.ones([2, 3])
y = tf.zeros([4], tf.int32)


def sink_list(c):
    assert len(c) == 2


def make():
    return tf.ones([4, 2])


class Holder:
    def load(self):
        self.items = make()


class Other:
    def build(self):
        self.items.append(1)


Alias = Other


class Both(Holder, Alias):
    pass


sink_list([x, y])
