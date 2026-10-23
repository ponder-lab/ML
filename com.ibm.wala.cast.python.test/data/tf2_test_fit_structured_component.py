# `fit` on a batched `tf.data.Dataset` built by `from_tensor_slices`, with Keras's default
# `train_step`, whose inputs component is itself a structure: a tuple, `((x, mask), y)`, or a dict,
# `({"ids": ids, "m": m}, y)`. The model's `call` then receives that structure, a tuple or a dict of
# tensors, and not a tensor; its members are tensors. With flat inputs, `(x, y)`, `call` receives a
# tensor.
import tensorflow as tf


def loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)


def consume_pair_inputs(inputs):
    assert isinstance(inputs, tuple) and len(inputs) == 2
    assert not isinstance(inputs, tf.Tensor)


def consume_pair_x(x):
    # Traced by `fit` with the batch axis unspecified.
    assert x.shape[1:] == (4,) and x.dtype == tf.float32


def consume_pair_mask(mask):
    assert mask.shape[1:] == (3,) and mask.dtype == tf.float32


def consume_pair_first(first):
    assert first.shape[1:] == (4,) and first.dtype == tf.float32


class Pair(tf.keras.Model):
    def call(self, inputs):
        consume_pair_inputs(inputs)
        consume_pair_first(inputs[0])
        x, mask = inputs
        consume_pair_x(x)
        consume_pair_mask(mask)
        return x


pair = Pair()
pair.compile(optimizer="sgd", loss=loss)
pair.fit(
    tf.data.Dataset.from_tensor_slices(
        ((tf.ones([8, 4]), tf.ones([8, 3])), tf.zeros([8]))
    ).batch(2),
    epochs=1,
    verbose=0,
)


def consume_mixed_inputs(inputs):
    assert isinstance(inputs, tuple) and len(inputs) == 2


def consume_mixed_ids(ids):
    assert ids.shape[1:] == (3,) and ids.dtype == tf.int32


class Mixed(tf.keras.Model):
    def call(self, inputs):
        consume_mixed_inputs(inputs)
        x, ids = inputs
        consume_mixed_ids(ids)
        return x


mixed = Mixed()
mixed.compile(optimizer="sgd", loss=loss)
mixed.fit(
    tf.data.Dataset.from_tensor_slices(
        ((tf.ones([8, 4]), tf.ones([8, 3], tf.int32)), tf.zeros([8]))
    ).batch(2),
    epochs=1,
    verbose=0,
)


def consume_keyed_inputs(inputs):
    assert isinstance(inputs, dict) and sorted(inputs) == ["ids", "m"]


def consume_keyed_ids(ids):
    assert ids.shape[1:] == (3,) and ids.dtype == tf.int32


def consume_keyed_m(m):
    assert m.shape[1:] == (5,) and m.dtype == tf.float32


class Keyed(tf.keras.Model):
    def call(self, inputs):
        consume_keyed_inputs(inputs)
        consume_keyed_ids(inputs["ids"])
        consume_keyed_m(inputs["m"])
        return inputs["m"]


keyed = Keyed()
keyed.compile(optimizer="sgd", loss=loss)
keyed.fit(
    tf.data.Dataset.from_tensor_slices(
        ({"ids": tf.ones([8, 3], tf.int32), "m": tf.ones([8, 5])}, tf.zeros([8]))
    ).batch(2),
    epochs=1,
    verbose=0,
)


def consume_flat_inputs(inputs):
    assert isinstance(inputs, tf.Tensor)
    assert inputs.shape[1:] == (4,) and inputs.dtype == tf.float32


class Flat(tf.keras.Model):
    def call(self, inputs):
        consume_flat_inputs(inputs)
        return inputs


flat = Flat()
flat.compile(optimizer="sgd", loss=loss)
flat.fit(
    tf.data.Dataset.from_tensor_slices((tf.ones([8, 4]), tf.zeros([8]))).batch(2),
    epochs=1,
    verbose=0,
)
