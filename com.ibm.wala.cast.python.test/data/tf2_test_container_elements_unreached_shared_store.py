# Test `ContainerElements` where a function the call graph does not reach resizes an attribute, `self.items`, under whose name an unreached method of another class stores a value a call returns, and a class has both classes as bases (wala/ML#1268).
# A `Both` is a `Holder` and an `Other`, so the value `Other.load` stores may be the one `Holder.build` appends to, and no list reports a length.
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
        self.items.append(1)


class Both(Holder, Other):
    pass


sink_list([x, y])
