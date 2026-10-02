# Test https://github.com/wala/ML/issues/997 and https://github.com/wala/ML/issues/188: a class
# call's keywords naming no formal of `__init__` reach its `**kwargs`, and its positional arguments
# past `__init__`'s formals reach its `*args`, as a call of a plain function's do.
import tensorflow as tf


def consume_kwargs_read(k):
    assert k.shape == (3, 3)
    return k


def consume_forwarded(k):
    assert k.shape == (3, 3)
    return k


def consume_keyword_only(t):
    assert t.shape == (7,)
    return t


def consume_keyword_only_second(t):
    assert t.shape == (6,)
    return t


def consume_second_positional(t):
    assert t.shape == (5,)
    return t


def target(kernel=None, **kw):
    consume_forwarded(kernel)


class Keyed:
    def __init__(self, behavior=None, **kwargs):
        consume_kwargs_read(kwargs["kernel"])
        target(**kwargs)


Keyed(kernel=tf.ones((3, 3)))


class Packed:
    def __init__(self, *args):
        consume_second_positional(args[1])


Packed(tf.ones(2), tf.ones(5))


class KeywordOnly:
    def __init__(self, *args, flag=None, **kw):
        consume_keyword_only(flag)
        consume_keyword_only_second(args[1])


KeywordOnly(tf.ones(2), tf.ones(6), flag=tf.ones(7))
