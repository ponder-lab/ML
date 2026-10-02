# Test https://github.com/wala/ML/issues/997: a field a base constructor writes to `self` when
# reached through `super().__init__(...)` lands on the instance under construction, so it is
# readable right after the write, after the super call, and in the instance's own methods.
import tensorflow as tf


def consume_in_base(p):
    assert p.shape == (3, 3)
    return p


def consume_after_super(p):
    assert p.shape == (3, 3)
    return p


def consume_in_call(p):
    assert p.shape == (3, 3)
    return p


class Base(tf.keras.Model):
    def __init__(self, stored=None, **kwargs):
        super().__init__(**kwargs)
        self.stored = stored
        consume_in_base(self.stored)

    def call(self, inputs):
        return inputs


class Net(Base):
    def __init__(self, **kwargs):
        super().__init__(stored=tf.ones((3, 3)), **kwargs)
        consume_after_super(self.stored)

    def call(self, inputs):
        consume_in_call(self.stored)
        return inputs


net = Net()
net(tf.ones((1, 4)))
