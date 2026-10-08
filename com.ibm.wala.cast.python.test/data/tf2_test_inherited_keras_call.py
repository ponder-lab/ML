# Test a Keras layer whose `call` is inherited from a user base class: calling an instance of the
# subclass reaches the base's `call`, as `Layer.__call__` finds `call` along the method resolution
# order, whether the call site sees only that class or several classes.
import tensorflow as tf


def consume_single(x):
    assert x.shape == (2, 3, 4)


def consume_own_init(x):
    assert x.shape == (2, 3, 4)


def consume_grandparent(x):
    assert x.shape == (2, 3, 4)


def consume_mixed(x):
    assert x.shape == (2, 3, 4)


def consume_override(x):
    assert x.shape == (2, 3, 4)


class SingleBase(tf.keras.layers.Layer):
    def call(self, inputs):
        consume_single(inputs)
        return inputs


class Single(SingleBase):
    pass


class OwnInitBase(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(OwnInitBase, self).__init__(**kwargs)
        self.filters = filters

    def call(self, inputs):
        consume_own_init(inputs)
        return inputs


class OwnInit(OwnInitBase):
    def __init__(self, filters, **kwargs):
        super().__init__(filters=filters, **kwargs)


class GrandBase(tf.keras.layers.Layer):
    def call(self, inputs):
        consume_grandparent(inputs)
        return inputs


class Middle(GrandBase):
    pass


class Grandchild(Middle):
    pass


class MixedBase(tf.keras.layers.Layer):
    def call(self, inputs):
        consume_mixed(inputs)
        return inputs


class Mixed(MixedBase):
    pass


class Stack(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.layers_list = [tf.keras.layers.Dense(4), Mixed(), tf.keras.layers.ReLU()]

    def call(self, x):
        for f in self.layers_list:
            x = f(x)
        return x


class OverrideBase(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs


class Override(OverrideBase):
    def call(self, inputs):
        consume_override(inputs)
        return inputs


x = tf.ones((2, 3, 4))
Single()(x)
OwnInit(4)(x)
Grandchild()(x)
Stack()(x)
Override()(x)
