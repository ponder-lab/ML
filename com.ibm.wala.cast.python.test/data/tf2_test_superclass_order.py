# Test https://github.com/wala/ML/issues/1014: a class with two program-defined bases has its first
# declared base as its superclass, and `super()` in it reaches the first base's method, on every run.
# Twelve such classes make a superclass picked in hash order, rather than declaration order, all but
# certain to be wrong for one of them.
import tensorflow as tf


def consume_first(x):
    assert x.shape == (2, 2)
    return x


class First0(object):
    def m(self):
        return tf.ones((2, 2))


class Second0(object):
    def m(self):
        return tf.ones((3, 3))


class Both0(First0, Second0):
    def m(self):
        return super().m()


class First1(object):
    def m(self):
        return tf.ones((2, 2))


class Second1(object):
    def m(self):
        return tf.ones((3, 3))


class Both1(First1, Second1):
    def m(self):
        return super().m()


class First2(object):
    def m(self):
        return tf.ones((2, 2))


class Second2(object):
    def m(self):
        return tf.ones((3, 3))


class Both2(First2, Second2):
    def m(self):
        return super().m()


class First3(object):
    def m(self):
        return tf.ones((2, 2))


class Second3(object):
    def m(self):
        return tf.ones((3, 3))


class Both3(First3, Second3):
    def m(self):
        return super().m()


class First4(object):
    def m(self):
        return tf.ones((2, 2))


class Second4(object):
    def m(self):
        return tf.ones((3, 3))


class Both4(First4, Second4):
    def m(self):
        return super().m()


class First5(object):
    def m(self):
        return tf.ones((2, 2))


class Second5(object):
    def m(self):
        return tf.ones((3, 3))


class Both5(First5, Second5):
    def m(self):
        return super().m()


class First6(object):
    def m(self):
        return tf.ones((2, 2))


class Second6(object):
    def m(self):
        return tf.ones((3, 3))


class Both6(First6, Second6):
    def m(self):
        return super().m()


class First7(object):
    def m(self):
        return tf.ones((2, 2))


class Second7(object):
    def m(self):
        return tf.ones((3, 3))


class Both7(First7, Second7):
    def m(self):
        return super().m()


class First8(object):
    def m(self):
        return tf.ones((2, 2))


class Second8(object):
    def m(self):
        return tf.ones((3, 3))


class Both8(First8, Second8):
    def m(self):
        return super().m()


class First9(object):
    def m(self):
        return tf.ones((2, 2))


class Second9(object):
    def m(self):
        return tf.ones((3, 3))


class Both9(First9, Second9):
    def m(self):
        return super().m()


class First10(object):
    def m(self):
        return tf.ones((2, 2))


class Second10(object):
    def m(self):
        return tf.ones((3, 3))


class Both10(First10, Second10):
    def m(self):
        return super().m()


class First11(object):
    def m(self):
        return tf.ones((2, 2))


class Second11(object):
    def m(self):
        return tf.ones((3, 3))


class Both11(First11, Second11):
    def m(self):
        return super().m()


for c in (
    Both0,
    Both1,
    Both2,
    Both3,
    Both4,
    Both5,
    Both6,
    Both7,
    Both8,
    Both9,
    Both10,
    Both11,
):
    consume_first(c().m())
