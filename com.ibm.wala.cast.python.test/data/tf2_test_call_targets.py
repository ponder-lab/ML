# Test `CallTargets`: each `call_*` function makes one call of interest, and the test reads the
# program functions it runs.
import tensorflow as tf


def helper(x):
    return x + 1


def call_direct(x):
    return helper(x)


class C:
    def m(self, x):
        return x * 2

    @staticmethod
    def s(x):
        return x

    @classmethod
    def k(cls, x):
        return x


def call_method(o, x):
    return o.m(x)


def call_bound(o, x):
    f = o.m
    return f(x)


def call_static(x):
    return C.s(x)


def call_class(x):
    return C.k(x)


def call_static_instance(o, x):
    return o.s(x)


class Sub(C):
    def call_super(self, x):
        return super().m(x)


def double(v):
    return v * 2


def call_map_fn(x):
    return tf.map_fn(double, x)


class MyLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs * 2


def call_layer(layer, x):
    return layer(x)


def call_mixed(c, x):
    f = helper if c else tf.identity
    return f(x)


def call_op(x):
    return tf.reshape(x, [-1])


x = tf.ones([2, 3])
o = C()
assert call_direct(1) == 2
assert call_method(o, 2) == 4
assert call_bound(o, 2) == 4
assert call_static(3) == 3
assert call_class(3) == 3
assert call_layer(MyLayer(), x).shape == (2, 3)
assert call_mixed(True, 1) == 2
assert call_mixed(False, x).shape == (2, 3)
assert call_op(x).shape == (6,)
assert call_static_instance(o, 3) == 3
assert Sub().call_super(2) == 4
assert call_map_fn(tf.ones([2])).shape == (2,)
