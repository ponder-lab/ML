# A class's decorators are applied as Python applies them, and its keywords are recorded on it
# (wala/ML#1260): the class object is passed to each decorator, innermost first, and the class's
# name is bound to what the outermost returns.
import abc
from dataclasses import dataclass

import tensorflow as tf

REGISTRY = {}


def consume_registered(x):
    assert x.shape == (2,)


def consume_replacement(x):
    assert x.shape == (3,)


def consume_stacked(x):
    assert x.shape == (4,)


def consume_dataclass(x):
    assert x.shape == (5,)


def consume_serializable(x):
    assert x.shape == (6,)


def consume_abstract(x):
    assert x.shape == (7,)


def register(cls):
    REGISTRY[cls.__name__] = cls
    return cls


def replace(cls):
    return Replacement


class Replacement:
    def __init__(self, x):
        consume_replacement(x)


class Stand:
    def __init__(self, x):
        consume_stacked(x)


def stand_in(cls):
    return Stand


@register
class Registered:
    def __init__(self, x):
        consume_registered(x)


@replace
class Replaced:
    def __init__(self, x):
        pass


# `stand_in` applies first and `register` second, so the name is bound to `Stand`.
@register
@stand_in
class Stacked:
    def __init__(self, x):
        pass


@dataclass
class Point:
    label: str

    def show(self, x):
        consume_dataclass(x)


@tf.keras.utils.register_keras_serializable(package="decorators")
class Serializable:
    def get_config(self):
        return {}

    def apply(self, x):
        consume_serializable(x)


class Abstract(metaclass=abc.ABCMeta):
    def run(self, x):
        consume_abstract(x)


Registered(tf.ones([2]))
Replaced(tf.ones([3]))
Stacked(tf.ones([4]))
Point("p").show(tf.ones([5]))
Serializable().apply(tf.ones([6]))
Abstract().run(tf.ones([7]))
assert REGISTRY == {"Registered": Registered, "Stand": Stand}
