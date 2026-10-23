# `fit` on a batched `tf.data.Dataset` built by `from_tensor_slices`, with Keras's default
# `train_step`, whose inputs are themselves a structure: a tuple, `((x, mask), y)`, or a dict,
# `({"ids": ids}, y)`. Each member of the inputs is sliced along its own leading axis and then
# batched, so the model's `call` receives one batch of each member.
import tensorflow as tf


def loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)


def consume_pair(inputs):
    assert isinstance(inputs, tuple) and len(inputs) == 2


def consume_pair_x(x):
    # Traced by `fit` with the batch axis unspecified.
    assert x.shape.rank == 2 and x.shape[1:] == (4,) and x.dtype == tf.float32


class Pair(tf.keras.Model):
    def call(self, inputs):
        consume_pair(inputs)
        x, mask = inputs
        consume_pair_x(x)
        return x


pair = Pair()
pair.compile(optimizer="sgd", loss=loss)
pair.fit(
    tf.data.Dataset.from_tensor_slices(
        ((tf.ones([8, 4]), tf.ones([8, 3])), tf.zeros([8], tf.int32))
    ).batch(2),
    epochs=1,
    verbose=0,
)


def consume_dict(inputs):
    assert isinstance(inputs, dict) and list(inputs) == ["ids"]


class Keyed(tf.keras.Model):
    def call(self, inputs):
        consume_dict(inputs)
        return tf.cast(inputs["ids"], tf.float32)


keyed = Keyed()
keyed.compile(optimizer="sgd", loss=loss)
keyed.fit(
    tf.data.Dataset.from_tensor_slices(
        ({"ids": tf.ones([8, 3], tf.int32)}, tf.zeros([8]))
    ).batch(4),
    epochs=1,
    verbose=0,
)
