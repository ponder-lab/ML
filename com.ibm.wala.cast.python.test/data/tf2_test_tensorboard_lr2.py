# Test https://github.com/wala/ML/issues/1341 through `from tensorflow.keras.callbacks import
# TensorBoard`, with the callback constructed from a `**` spread: the callback calls the schedule
# with the optimizer's `int64` step count, as `tf2_test_tensorboard_lr.py` checks for
# `tf.keras.callbacks.TensorBoard`.
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

seen = set()


class Imported(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __call__(self, step):
        assert step.shape == ()
        seen.add(step.dtype)
        return 0.1


class Net(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(3)

    def call(self, inputs):
        return self.dense(inputs)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(y - self(x, training=True)))
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}


config = {"log_dir": tempfile.mkdtemp()}
net = Net()
net.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=Imported()))
net.fit(
    np.ones((4, 2), dtype=np.float32),
    np.ones((4, 3), dtype=np.float32),
    epochs=1,
    verbose=0,
    callbacks=[TensorBoard(**config)],
)
assert seen == {tf.int64, tf.float32}, seen
