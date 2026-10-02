# Test https://github.com/wala/ML/issues/997: `fit` on a dataset whose element is a tuple of a
# dict, the targets and the sample weights, unpacked in `train_step` by
# `tf.keras.utils.unpack_x_y_sample_weight`; and `fit` on a dataset of single tensors, where the
# unpacked inputs are the whole element.
import numpy as np
import tensorflow as tf


def consume_targets(y):
    assert y.shape[1:] == (3,) and y.dtype == tf.float32
    return y


def consume_weights(w):
    assert w.shape[1:] == () and w.dtype == tf.float32
    return w


def consume_single_inputs(x):
    assert x.shape[1:] == (4,) and x.dtype == tf.float32
    return x


def consume_single_call(inputs):
    assert inputs.shape[1:] == (4,) and inputs.dtype == tf.float32
    return inputs


class Psych(tf.keras.Model):
    # The dict component is the model's input; its own subscript is read in `call`.
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(3)

    def call(self, inputs):
        return self.dense(inputs["stimulus_set"])

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        consume_targets(y)
        consume_weights(sample_weight)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(
                tf.square(self(x, training=True) - y) * sample_weight[:, None]
            )
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}


class Single(tf.keras.Model):
    # A dataset of single tensors: the unpacked inputs are the whole element.
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(4)

    def call(self, inputs):
        consume_single_call(inputs)
        return self.dense(inputs)

    def train_step(self, data):
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        consume_single_inputs(x)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(self(x, training=True) - x))
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}


features = {
    "stimulus_set": np.ones((8, 4), dtype=np.float32),
    "groups": np.ones((8, 2), dtype=np.float32),
}
tuple_dataset = tf.data.Dataset.from_tensor_slices(
    (features, np.ones((8, 3), dtype=np.float32), np.ones((8,), dtype=np.float32))
).batch(2)
psych = Psych()
psych.compile(optimizer="sgd", loss="mse")
psych.fit(tuple_dataset, epochs=1, verbose=0)

single_dataset = tf.data.Dataset.from_tensor_slices(
    np.ones((8, 4), dtype=np.float32)
).batch(2)
single = Single()
single.compile(optimizer="sgd", loss="mse")
single.fit(single_dataset, epochs=1, verbose=0)
