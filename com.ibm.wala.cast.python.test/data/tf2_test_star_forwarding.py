# Test https://github.com/wala/ML/issues/991: a wrapper whose formals are `*args` and `**kwargs`
# forwards every positional and keyword argument to the function it calls.
import tensorflow as tf


def two_positionals(x, y):
    assert x.shape == (2,) and y.shape == (3,)
    return y


def keyword(x, scale=None):
    assert scale.shape == (4,)
    return scale


def mixed(x, y, scale=None):
    assert y.shape == (3,) and scale.shape == (4,)
    return scale


def method_target(x, y):
    assert x.shape == (2,) and y.shape == (3,)
    return y


def forwarded_target(x, y):
    assert x.shape == (2,) and y.shape == (3,)
    return y


def literal_kw(x, scale=None):
    assert scale.shape == (4,)
    return scale


def local_kw(x, scale=None):
    assert scale.shape == (4,)
    return scale


def make_kw():
    return {"scale": tf.ones(4)}


class Built:
    def __init__(self, x, y):
        assert x.shape == (2,) and y.shape == (3,)
        self.x = x
        self.y = y


class Holder:
    def collect(self, *args):
        # A method's `*args` packs the instance call's arguments.
        return method_target(*args)

    def forward(self, x, y):
        return forwarded_target(x, y)

    def take(self, x, scale=None):
        assert scale.shape == (4,)
        return scale


def wrap(fn):
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


wrap(two_positionals)(tf.ones(2), tf.ones(3))
wrap(keyword)(tf.ones(2), scale=tf.ones(4))
wrap(mixed)(tf.ones(2), tf.ones(3), scale=tf.ones(4))
Holder().collect(tf.ones(2), tf.ones(3))
pair = [tf.ones(2), tf.ones(3)]
Holder().forward(*pair)
literal_kw(tf.ones(2), **{"scale": tf.ones(4)})
options = {"scale": tf.ones(4)}
local_kw(tf.ones(2), **options)
Holder().take(tf.ones(2), **make_kw())
Built(*pair)
