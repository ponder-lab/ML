# Test https://github.com/wala/ML/issues/997: `fit` on a `tf.data.Dataset` feeds the model the
# dataset's element, so a `train_step` override sees the batch's inputs and targets, and the model's
# `call` sees exactly the inputs.
import numpy as np
import tensorflow as tf


def consume_ds_x(x):
    assert x.shape[1:] == (4,) and x.dtype == tf.float32
    return x


def consume_ds_y(y):
    assert y.shape[1:] == (3,) and y.dtype == tf.float32
    return y


def consume_call_inputs(inputs):
    assert inputs.shape[1:] == (4,) and inputs.dtype == tf.float32
    return inputs


class Net(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(3)

    def call(self, inputs):
        consume_call_inputs(inputs)
        return self.dense(inputs)

    def train_step(self, data):
        x, y = data
        consume_ds_x(x)
        consume_ds_y(y)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(self(x, training=True) - y))
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}


dataset = tf.data.Dataset.from_tensor_slices(
    (np.ones((8, 4), dtype=np.float32), np.ones((8, 3), dtype=np.float32))
).batch(2)
net = Net()
net.compile(optimizer="sgd", loss="mse")
net.fit(dataset, epochs=1, verbose=0)
