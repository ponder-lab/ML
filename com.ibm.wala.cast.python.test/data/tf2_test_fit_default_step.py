# Test https://github.com/wala/ML/issues/997: `fit` on a `tf.data.Dataset` with Keras's default
# `train_step` feeds the model's `call` the element's first component, the inputs, and not the
# whole element, targets included. When the inputs are themselves a tuple, `call` reads within
# them.
import tensorflow as tf


def flat_generator():
    yield tf.ones([3, 4]), tf.zeros([3], tf.int32)


def nested_generator():
    yield (
        tf.ones([3, 4]),
        tf.zeros([2, 5], tf.int32),
        tf.zeros([3], tf.int32),
    ), tf.zeros([1], tf.int32)


def nested_dataset(context=None):
    return tf.data.Dataset.from_generator(
        nested_generator,
        output_types=((tf.float32, tf.int32, tf.int32), tf.int32),
        output_shapes=(
            (
                tf.TensorShape([None, 4]),
                tf.TensorShape([2, None]),
                tf.TensorShape([None]),
            ),
            tf.TensorShape([None]),
        ),
    )


def consume_flat_inputs(inputs):
    assert inputs.shape[1:] == (4,) and inputs.dtype == tf.float32


def consume_nested_index(index):
    assert index.shape.rank == 1 and index.dtype == tf.int32


def consume_distributed_index(index):
    assert index.shape.rank == 1 and index.dtype == tf.int32


def consume_override_index(index):
    assert index.shape.rank == 1 and index.dtype == tf.int32


def loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)


class Flat(tf.keras.Model):
    def call(self, inputs, training=None, mask=None):
        consume_flat_inputs(inputs)
        return tf.reduce_sum(inputs, axis=0, keepdims=True)


class Nested(tf.keras.Model):
    def call(self, inputs, training=None, mask=None):
        x, edge_index, node_graph_index = inputs
        consume_nested_index(node_graph_index)
        return tf.reduce_sum(x, axis=0, keepdims=True)


class Distributed(tf.keras.Model):
    def call(self, inputs, training=None, mask=None):
        x, edge_index, node_graph_index = inputs
        consume_distributed_index(node_graph_index)
        return tf.reduce_sum(x, axis=0, keepdims=True)


class Override(tf.keras.Model):
    def call(self, inputs, training=None, mask=None):
        x, edge_index, node_graph_index = inputs
        consume_override_index(node_graph_index)
        return tf.reduce_sum(x, axis=0, keepdims=True)

    def train_step(self, data):
        x, y = data
        self(x, training=True)
        return {}


flat = Flat()
flat.compile(optimizer="sgd", loss=loss)
flat.fit(
    tf.data.Dataset.from_generator(
        flat_generator,
        output_types=(tf.float32, tf.int32),
        output_shapes=(tf.TensorShape([None, 4]), tf.TensorShape([None])),
    ),
    epochs=1,
    verbose=0,
)

nested = Nested()
nested.compile(optimizer="sgd", loss=loss)
nested.fit(nested_dataset(), epochs=1, verbose=0)

strategy = tf.distribute.MirroredStrategy()
distributed_dataset = strategy.experimental_distribute_datasets_from_function(
    nested_dataset
)
with strategy.scope():
    distributed = Distributed()
distributed.compile(optimizer="sgd", loss=loss)
distributed.fit(distributed_dataset, steps_per_epoch=1, epochs=1, verbose=0)

override = Override()
override.compile(optimizer="sgd", loss=loss)
override.fit(nested_dataset(), epochs=1, verbose=0)
