# Fixture for wala/ML#1048: a class without its own `__init__` runs the one it inherits.
#
# Each subclass below defines no `__init__`, so constructing it runs the nearest one in its method
# resolution order, and the field that initializer sets holds the constructor's argument.
import numpy as np
import tensorflow as tf


def consume(x):
    pass


def consume_object_base(x):
    pass


def consume_second_base(x):
    pass


def consume_grandparent(x):
    pass


def consume_keras(x):
    pass


class Base:
    def __init__(self, x):
        self.x = x


class Sub(Base):
    pass


class ObjectBase(object):
    def __init__(self, x):
        self.x = x


class ObjectSub(ObjectBase):
    def other(self):
        return self.x


class Plain:
    pass


class SecondBase:
    def __init__(self, y):
        self.y = y


class Mixed(Plain, SecondBase):
    pass


class Middle(Base):
    pass


class Leaf(Middle):
    pass


class KerasBase(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units


class KerasSub(KerasBase):
    pass


sub = Sub(np.zeros([3, 4], dtype=np.float32))
assert sub.x.shape == (3, 4)
consume(sub.x)

object_sub = ObjectSub(np.zeros([5], dtype=np.float32))
assert object_sub.x.shape == (5,)
consume_object_base(object_sub.x)

mixed = Mixed(np.zeros([2, 2], dtype=np.int32))
assert mixed.y.shape == (2, 2)
consume_second_base(mixed.y)

leaf = Leaf(np.zeros([7], dtype=np.float32))
assert leaf.x.shape == (7,)
consume_grandparent(leaf.x)

keras_sub = KerasSub(tf.ones([6]))
assert keras_sub.units.shape == (6,)
consume_keras(keras_sub.units)
