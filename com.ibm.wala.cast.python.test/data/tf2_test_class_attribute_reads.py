# Test reads of class-body attributes through an instance, beside the cases that must not change.
import tensorflow as tf


def shadow_sink(x):
    return x


def override_sink(x):
    return x


def class_read_sink(x):
    return x


def config_sink(x):
    return x


def inherited_sink(x):
    return x


class Shadow:
    """A method the instance shadows with an attribute."""

    def value(self):
        return 1

    def __init__(self):
        self.value = tf.ones([2])

    def use(self):
        return shadow_sink(self.value)


class Override:
    """A class attribute an instance write overrides."""

    scale = tf.ones([3])

    def __init__(self):
        self.scale = tf.ones([2])

    def use(self):
        return override_sink(self.scale)


class Constants:
    """A class attribute read through the class itself."""

    base = tf.ones([4])


class Config:
    """Class attributes that are not tensors, read through the instance."""

    size = 3

    def make(self):
        return config_sink(tf.ones([self.size]))


class Base:
    """A class attribute an instance of a subclass reads."""

    table = tf.ones([5])


class Derived(Base):
    def use(self):
        return inherited_sink(self.table)


def alias_sink(x):
    return x


def lambda_sink(x):
    return x


def property_sink(x):
    return x


def compiled_sink(x):
    return x


def _step(self, x):
    compiled_sink(self)
    return x


def scaled(self, x):
    return alias_sink(self)


def getter(self):
    return tf.ones([4])


class Bound:
    """Class-body attributes bound to functions and a property: Python binds them on the read."""

    act = scaled
    f = lambda self, x: lambda_sink(self)
    p = property(getter)
    step = tf.function(_step)

    def use(self, t):
        self.act(t)
        self.f(t)
        self.step(t)
        return property_sink(self.p)


alias_sink(None)
lambda_sink(None)
compiled_sink(None)
assert Bound().use(tf.ones([2])).shape == (4,)

assert Shadow().use().shape == (2,)
assert Override().use().shape == (2,)
assert class_read_sink(Constants.base).shape == (4,)
assert Config().make().shape == (3,)
assert Derived().use().shape == (5,)
