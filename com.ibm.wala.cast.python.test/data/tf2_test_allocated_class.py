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


class ProjectMixin:

    def project(self, x):
        return x


class MixedNet(ProjectMixin, tf.keras.Model):

    def call(self, x):
        return self.project(x)


class MixedNetModelFirst(tf.keras.Model, ProjectMixin):

    def call(self, x):
        return self.project(x)


class Inner(ProjectMixin, tf.keras.layers.Layer):

    def call(self, x):
        return self.project(x)


class Outer(Inner):
    pass


class MixedPlain(ProjectMixin, Plain):
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


def consume_method(v):
    return v


def consume_mixed_net(v):
    return v


def consume_mixed_net_model_first(v):
    return v


def consume_outer(v):
    return v


def consume_mixed_plain(v):
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
ds = tf.data.Dataset.range(3)
assert callable(consume_method(ds.shuffle))
assert isinstance(consume_mixed_net(MixedNet()), tf.keras.Model)
assert isinstance(consume_mixed_net_model_first(MixedNetModelFirst()), tf.keras.Model)
outer = consume_outer(Outer())
assert isinstance(outer, tf.keras.layers.Layer) and not isinstance(
    outer, tf.keras.Model
)
mixed_plain = consume_mixed_plain(MixedPlain(3))
assert not isinstance(mixed_plain, (tf.keras.layers.Layer, tf.keras.Model))
