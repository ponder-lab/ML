# Test https://github.com/wala/ML/issues/997: `fit`, `evaluate` and `predict` on a
# `tf.data.Dataset` give the model's step the dataset's element, so each member of a structured
# component keeps its own shape and dtype, and a value read of `call`'s inputs sees the component
# the step selected, never the targets.
import tensorflow as tf


def consume_mixed_x(x):
    pass


def consume_mixed_ids(ids):
    pass


def consume_mixed_inputs(inputs):
    pass


def consume_same_x(x):
    pass


def consume_same_mask(mask):
    pass


def consume_short(inputs):
    pass


def consume_first(inputs):
    pass


def consume_second(inputs):
    pass


def consume_predicted(inputs):
    pass


def consume_dict_a(a):
    pass


def consume_dict_b(b):
    pass


def consume_padded(inputs):
    pass


def consume_scaled(scaled):
    pass


def loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)


class Mixed(tf.keras.Model):
    """A component that is a pair of a float and an int tensor (wala/ML#1138)."""

    def call(self, inputs, training=None, mask=None):
        consume_mixed_inputs(inputs)
        x, ids = inputs
        assert x.shape == (2, 4) and x.dtype == tf.float32
        consume_mixed_x(x)
        assert ids.shape == (2, 3) and ids.dtype == tf.int32
        consume_mixed_ids(ids)
        return tf.reduce_sum(x, axis=1, keepdims=True)


mixed = Mixed()
mixed.compile(optimizer="sgd", loss=loss, run_eagerly=True)
mixed.fit(
    tf.data.Dataset.from_tensor_slices(
        ((tf.ones([8, 4]), tf.zeros([8, 3], tf.int32)), tf.zeros([8], tf.int32))
    ).batch(2),
    epochs=1,
    verbose=0,
)


class Same(tf.keras.Model):
    """A component that is a pair of float tensors of different shapes (wala/ML#1138)."""

    def call(self, inputs, training=None, mask=None):
        x, mask_ = inputs
        assert x.shape == (2, 4) and x.dtype == tf.float32
        consume_same_x(x)
        assert mask_.shape == (2, 3) and mask_.dtype == tf.float32
        consume_same_mask(mask_)
        return tf.reduce_sum(x, axis=1, keepdims=True)


same = Same()
same.compile(optimizer="sgd", loss=loss, run_eagerly=True)
same.fit(
    tf.data.Dataset.from_tensor_slices(
        ((tf.ones([8, 4]), tf.ones([8, 3])), tf.zeros([8]))
    ).batch(2),
    epochs=1,
    verbose=0,
)


class Short(tf.keras.Model):
    """Five rows batched by two: the last batch has one row."""

    def call(self, inputs, training=None, mask=None):
        assert inputs.shape in ((2, 4), (1, 4)) and inputs.dtype == tf.float32
        consume_short(inputs)
        return tf.reduce_sum(inputs, axis=1, keepdims=True)


short = Short()
short.compile(optimizer="sgd", loss=loss, run_eagerly=True)
short.fit(
    tf.data.Dataset.from_tensor_slices((tf.ones([5, 4]), tf.zeros([5]))).batch(2),
    epochs=1,
    verbose=0,
)


class First(tf.keras.Model):
    """One of two models fit on datasets whose elements disagree."""

    def call(self, inputs, training=None, mask=None):
        assert inputs.shape == (2, 4) and inputs.dtype == tf.float32
        consume_first(inputs)
        return tf.reduce_sum(inputs, axis=1, keepdims=True)


class Second(tf.keras.Model):
    """The other of the two models."""

    def call(self, inputs, training=None, mask=None):
        assert inputs.shape == (3, 6) and inputs.dtype == tf.int32
        consume_second(inputs)
        return tf.reduce_sum(tf.cast(inputs, tf.float32), axis=1, keepdims=True)


first = First()
first.compile(optimizer="sgd", loss=loss, run_eagerly=True)
first.fit(
    tf.data.Dataset.from_tensor_slices((tf.ones([8, 4]), tf.zeros([8]))).batch(2),
    epochs=1,
    verbose=0,
)
second = Second()
second.compile(optimizer="sgd", loss=loss, run_eagerly=True)
second.fit(
    tf.data.Dataset.from_tensor_slices(
        (tf.zeros([9, 6], tf.int32), tf.zeros([9]))
    ).batch(3),
    epochs=1,
    verbose=0,
)


class Predicted(tf.keras.Model):
    """`predict` on a dataset of inputs alone."""

    def call(self, inputs, training=None, mask=None):
        assert inputs.shape == (2, 5) and inputs.dtype == tf.float32
        consume_predicted(inputs)
        return tf.reduce_sum(inputs, axis=1, keepdims=True)


predicted = Predicted()
predicted.compile(run_eagerly=True)
predicted.predict(
    tf.data.Dataset.from_tensor_slices(tf.ones([8, 5])).batch(2), verbose=0
)


class Keyed(tf.keras.Model):
    """A component that is a dict of tensors."""

    def call(self, inputs, training=None, mask=None):
        a = inputs["a"]
        assert a.shape == (2, 4) and a.dtype == tf.float32
        consume_dict_a(a)
        b = inputs["b"]
        assert b.shape == (2, 2) and b.dtype == tf.int64
        consume_dict_b(b)
        return tf.reduce_sum(a, axis=1, keepdims=True)


keyed = Keyed()
keyed.compile(optimizer="sgd", loss=loss, run_eagerly=True)
keyed.evaluate(
    tf.data.Dataset.from_tensor_slices(
        ({"a": tf.ones([8, 4]), "b": tf.zeros([8, 2], tf.int64)}, tf.zeros([8]))
    ).batch(2),
    verbose=0,
)


class Padded(tf.keras.Model):
    """A `padded_batch` of fixed-length rows."""

    def call(self, inputs, training=None, mask=None):
        assert inputs.shape == (4, 3) and inputs.dtype == tf.float32
        consume_padded(inputs)
        return tf.reduce_sum(inputs, axis=1, keepdims=True)


padded = Padded()
padded.compile(optimizer="sgd", loss=loss, run_eagerly=True)
padded.fit(
    tf.data.Dataset.from_tensor_slices((tf.ones([8, 3]), tf.zeros([8]))).padded_batch(
        4
    ),
    epochs=1,
    verbose=0,
)


class Scaled(tf.keras.Model):
    """A value read of the inputs, beside the targets of another shape (wala/ML#1141)."""

    def call(self, inputs, training=None, mask=None):
        scaled = inputs * 2.0
        assert scaled.shape == (2, 6) and scaled.dtype == tf.float32
        consume_scaled(scaled)
        return tf.reduce_sum(inputs, axis=1, keepdims=True)


scaled_model = Scaled()
scaled_model.compile(optimizer="sgd", loss=loss, run_eagerly=True)
scaled_model.fit(
    tf.data.Dataset.from_tensor_slices((tf.ones([8, 6]), tf.zeros([8, 5]))).batch(2),
    epochs=1,
    verbose=0,
)
