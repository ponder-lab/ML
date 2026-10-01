# Test https://github.com/wala/ML/issues/188: a bare decorator `@d` is applied as `d(f)`, so a
# decorator's wrapper reaches the function it decorates.
import functools
from functools import wraps

import tensorflow as tf


def ident(function):
    return function


def plain(function):
    def wrap(image):
        return function(image)

    return wrap


def qualified(function):
    @functools.wraps(function)
    def wrap(image, *args, **kwargs):
        return function(image, **kwargs)

    return wrap


def bare(function):
    @wraps(function)
    def wrap(image, *args, **kwargs):
        return function(image, **kwargs)

    return wrap


def outer(function):
    def wrap(image):
        return function(tf.ones((3,)))

    return wrap


def inner(function):
    def wrap(image):
        return function(tf.cast(image, tf.int32))

    return wrap


class Wrapper:
    def __init__(self, function):
        self.function = function

    def __call__(self, image):
        return self.function(image)


@ident
def f_ident(image):
    assert image.shape == (2,) and image.dtype == tf.float32
    return image


@plain
def f_plain(image):
    assert image.shape == (2,) and image.dtype == tf.float32
    return image


@qualified
def f_qualified(image):
    assert image.shape == (2,) and image.dtype == tf.float32
    return image


@bare
def f_bare(image):
    assert image.shape == (2,) and image.dtype == tf.float32
    return image


# Decorators apply from the `def` outward: `outer(inner(f_stacked))`, so `f_stacked` receives
# `inner`'s cast of `outer`'s ones, an int32 tensor of shape (3,).
@outer
@inner
def f_stacked(image):
    assert image.shape == (3,) and image.dtype == tf.int32
    return image


@Wrapper
def f_class(image):
    assert image.shape == (2,) and image.dtype == tf.float32
    return image


f_ident(tf.ones(2))
f_plain(tf.ones(2))
f_qualified(tf.ones(2))
f_bare(tf.ones(2))
f_stacked(tf.ones(2))
f_class(tf.ones(2))
