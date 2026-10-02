# Test https://github.com/wala/ML/issues/1006: a class with several program-defined bases, a Keras
# model or a plain base plus two mixins, reaches the methods of every base when they are called on
# its instance, including a mixin method that calls another method of its mixin through `self`.
import tensorflow as tf


def consume_second_base(x):
    assert x.shape == (2, 2)
    return x


def consume_third_base(x):
    assert x.shape == (3, 3)
    return x


def consume_third_via_helper(x):
    assert x.shape == (4, 4)
    return x


def consume_plain_third(x):
    assert x.shape == (5, 5)
    return x


class FirstMixin(object):
    def first(self, x):
        return x


class SecondMixin(object):
    def second(self, x):
        return self._helper(x)

    def _helper(self, x):
        consume_third_via_helper(tf.ones((4, 4)))
        return x


class Detector(tf.keras.Model, FirstMixin, SecondMixin):
    def __init__(self, n, **kwargs):
        super(Detector, self).__init__(**kwargs)
        self.n = n

    def __call__(self, inputs, training=True):
        return inputs


class PlainBase(object):
    def __init__(self, n):
        self.n = n


class PlainDetector(PlainBase, FirstMixin, SecondMixin):
    def __init__(self, n):
        super(PlainDetector, self).__init__(n)


model = Detector(3)
model(tf.ones((1, 1)), training=False)
consume_second_base(model.first(tf.ones((2, 2))))
consume_third_base(model.second(tf.ones((3, 3))))
consume_plain_third(PlainDetector(3).second(tf.ones((5, 5))))


def consume_diamond(x):
    assert x.shape == (6, 6)
    return x


def consume_shadowed(x):
    return x


class DiamondA(object):
    def m(self, x):
        consume_shadowed(x)
        return x


class DiamondB(DiamondA):
    def m(self, x):
        consume_diamond(x)
        return x


class DiamondC(DiamondA):
    pass


class DiamondD(DiamondC, DiamondB):
    pass


# Python's method resolution order is D, C, B, A, so `m` is `DiamondB.m`.
DiamondD().m(tf.ones((6, 6)))
consume_shadowed(1)
