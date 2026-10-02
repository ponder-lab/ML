# Test https://github.com/wala/ML/issues/997: a `train_step` reached both through `fit` and by a
# direct call with the user's own batch sees the inputs of both calls.
import numpy as np
import tensorflow as tf


def consume_both(x):
    assert x.shape[1:] in ((4,), (6,)) and x.dtype == tf.float32
    return x


class Both(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(3)

    def call(self, inputs):
        return self.dense(inputs)

    def train_step(self, data):
        x, y = data
        consume_both(x)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(self(x, training=True) - y))
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}


model = Both()
model.compile(optimizer="sgd", loss="mse")
model.fit(
    np.ones((8, 4), dtype=np.float32),
    np.ones((8, 3), dtype=np.float32),
    epochs=1,
    verbose=0,
)
# The same step, called directly with the user's own batch of a different width.
direct = Both()
direct.compile(optimizer="sgd", loss="mse")
direct.train_step((tf.ones((5, 6)), tf.ones((5, 3))))
