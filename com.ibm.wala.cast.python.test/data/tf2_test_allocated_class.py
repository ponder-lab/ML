# The class whose constructor allocated each value a sink receives (wala/ML#1165).
import tensorflow as tf


class Encoder(tf.keras.layers.Layer):

    def call(self, x):
        return x


class Holder(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()

    def call(self, x):
        consume_encoder(self.encoder)
        return self.encoder(x)


class Plain:

    def __init__(self, n):
        self.n = n


class Base(tf.keras.layers.Layer):

    def call(self, x):
        return x


class Derived(Base):
    pass


def consume_encoder(v):
    return v


def consume_holder(v):
    return v


def consume_plain(v):
    return v


def consume_derived(v):
    return v


def consume_dense(v):
    return v


def consume_constant(v):
    return v


def consume_length(v):
    return v


def consume_class(v):
    return v


def consume_callable(v):
    return v


xs = [1, 2, 3]
holder = Holder()
holder(tf.ones([1, 2]))
assert isinstance(consume_holder(holder), tf.keras.Model)
assert isinstance(holder.encoder, tf.keras.layers.Layer)
assert not isinstance(holder.encoder, tf.keras.Model)
assert type(consume_plain(Plain(3))) is Plain
derived = consume_derived(Derived())
assert isinstance(derived, Base) and isinstance(derived, tf.keras.layers.Layer)
assert isinstance(consume_dense(tf.keras.layers.Dense(4)), tf.keras.layers.Layer)
assert consume_constant(3) == 3
assert consume_length(len(xs)) == 3
assert consume_class(Plain) is Plain
assert consume_callable(tf.keras.layers.Dense) is tf.keras.layers.Dense
