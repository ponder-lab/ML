# Test for a `tf.keras.Model` subclass that builds a functional model in its constructor and does
# not override `call`: applying an instance runs `Model.call` over the graph from `inputs` to
# `outputs`.
import tensorflow as tf


class FunctionalKeyword(tf.keras.Model):

    def __init__(self):
        i = tf.keras.Input(shape=(3,))
        super().__init__(inputs=i, outputs=tf.keras.layers.Dense(4)(i))


class FunctionalPositional(tf.keras.Model):

    def __init__(self):
        i = tf.keras.Input(shape=(3,))
        super().__init__(i, tf.keras.layers.Dense(4)(i))


class NotFunctional(tf.keras.Model):
    pass


class InheritedInit(tf.keras.Model):
    pass


class KeywordsForwarded(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class PositionalsForwarded(tf.keras.Model):

    def __init__(self, *args):
        super().__init__(*args)


def consume_keyword(t):
    assert t.shape == (2, 4) and t.dtype == tf.float32


def consume_positional(t):
    assert t.shape == (2, 4) and t.dtype == tf.float32


def consume_direct(t):
    assert t.shape == (2, 4) and t.dtype == tf.float32


def consume_call(t):
    assert t.shape == (2, 4) and t.dtype == tf.float32


def consume_inherited_keyword(t):
    assert t.shape == (2, 4) and t.dtype == tf.float32


def consume_inherited_positional(t):
    assert t.shape == (2, 4) and t.dtype == tf.float32


def consume_keywords_forwarded(t):
    assert t.shape == (2, 4) and t.dtype == tf.float32


def consume_positionals_forwarded(t):
    assert t.shape == (2, 4) and t.dtype == tf.float32


def consume_not_functional(t):
    raise AssertionError("Not reached: the call raises.")


consume_keyword(FunctionalKeyword()(tf.ones([2, 3])))
consume_positional(FunctionalPositional()(tf.ones([2, 3])))

i = tf.keras.Input(shape=(3,))
consume_direct(
    tf.keras.Model(inputs=i, outputs=tf.keras.layers.Dense(4)(i))(tf.ones([2, 3]))
)

direct = tf.keras.Model(inputs=i, outputs=tf.keras.layers.Dense(4)(i))
consume_call(direct.call(tf.ones([2, 3])))

j = tf.keras.Input(shape=(3,))
consume_inherited_keyword(
    InheritedInit(inputs=j, outputs=tf.keras.layers.Dense(4)(j))(tf.ones([2, 3]))
)
consume_inherited_positional(
    InheritedInit(j, tf.keras.layers.Dense(4)(j))(tf.ones([2, 3]))
)
consume_keywords_forwarded(
    KeywordsForwarded(inputs=j, outputs=tf.keras.layers.Dense(4)(j))(tf.ones([2, 3]))
)
consume_positionals_forwarded(
    PositionalsForwarded(j, tf.keras.layers.Dense(4)(j))(tf.ones([2, 3]))
)

try:
    consume_not_functional(NotFunctional()(tf.ones([2, 3])))
except NotImplementedError:
    pass
