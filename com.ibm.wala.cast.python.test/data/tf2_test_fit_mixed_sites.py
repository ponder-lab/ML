# Test https://github.com/wala/ML/issues/997: the inputs a model is fit on are a tensor on one path
# and a dataset on the other, so the one `fit` call's packed data holds both, and the step sees
# both calls' inputs.
import numpy as np
import tensorflow as tf


def consume_mixed(x):
    assert x.shape[1:] == (4,) and x.dtype == tf.float32
    return x


class Net(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(3)

    def call(self, inputs):
        return self.dense(inputs)

    def train_step(self, data):
        x, y = data
        consume_mixed(x)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(self(x, training=True) - y))
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}


dataset = tf.data.Dataset.from_tensor_slices(
    (np.ones((6, 4), dtype=np.float32), np.ones((6, 3), dtype=np.float32))
).batch(2)


def inputs(use_dataset):
    if use_dataset:
        return dataset, None
    return np.ones((8, 4), dtype=np.float32), np.ones((8, 3), dtype=np.float32)


for use_dataset in (False, True):
    x, y = inputs(use_dataset)
    model = Net()
    model.compile(optimizer="sgd", loss="mse")
    model.fit(x, y, epochs=1, verbose=0)
