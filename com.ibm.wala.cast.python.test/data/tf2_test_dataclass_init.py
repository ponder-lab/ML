# The initializer `@dataclass` generates binds its arguments to the class's annotated fields, in
# declaration order, and keyword arguments by name (wala/ML#205).
import dataclasses
from dataclasses import dataclass as record

import tensorflow as tf


def consume_first(x):
    assert x.shape == (2,)


def consume_second(x):
    assert x.shape == (3, 3)


def consume_keyword(x):
    assert x.shape == (4,)


def consume_aliased(x):
    assert x.shape == (5,)


def consume_default(x):
    assert x == 0


@dataclasses.dataclass
class Pair:
    first: object
    second: object
    scale: int = 0

    def use(self):
        consume_first(self.first)
        consume_second(self.second)
        consume_default(self.scale)


@dataclasses.dataclass(frozen=True)
class Keyworded:
    value: object

    def use(self):
        consume_keyword(self.value)


@record
class Aliased:
    value: object

    def use(self):
        consume_aliased(self.value)


Pair(tf.ones([2]), tf.ones([3, 3])).use()
Keyworded(value=tf.ones([4])).use()
Aliased(tf.ones([5])).use()
