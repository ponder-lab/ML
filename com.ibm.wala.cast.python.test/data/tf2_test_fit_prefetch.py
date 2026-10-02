# Test https://github.com/wala/ML/issues/997: a model fit on a dataset whose pipeline ends in a
# pass-through transformation (`shuffle`, `prefetch`) has its step's inputs read as the batched
# element's first component, through the transformations to the batch.
import numpy as np
import tensorflow as tf


def consume_prefetched(x):
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
        consume_prefetched(x)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(self(x, training=True) - y))
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}


dataset = (
    tf.data.Dataset.from_tensor_slices(
        (np.ones((6, 4), dtype=np.float32), np.ones((6, 3), dtype=np.float32))
    )
    .shuffle(6)
    .batch(2)
    .prefetch(1)
)
model = Net()
model.compile(optimizer="sgd", loss="mse")
model.fit(dataset, epochs=1, verbose=0)
