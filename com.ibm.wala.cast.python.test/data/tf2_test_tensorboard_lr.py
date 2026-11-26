# Test https://github.com/wala/ML/issues/1341: a Keras `TensorBoard` callback logs the learning
# rate at each epoch's end by calling the optimizer's schedule with the optimizer's `int64` step
# count, `iterations`, as it is, while applying gradients calls it with the count cast to `float32`.
import tempfile

import numpy as np
import tensorflow as tf

seen = set()


class Schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __call__(self, step):
        assert step.shape == ()
        seen.add(step.dtype)
        return 0.1


class Untracked(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __call__(self, step):
        assert step.shape == () and step.dtype == tf.float32
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


x = np.ones((4, 2), dtype=np.float32)
y = np.ones((4, 3), dtype=np.float32)

net = Net()
net.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=Schedule()))
net.fit(
    x,
    y,
    epochs=1,
    verbose=0,
    callbacks=[tf.keras.callbacks.TensorBoard(log_dir=tempfile.mkdtemp())],
)
assert seen == {tf.int64, tf.float32}, seen

# Without a `TensorBoard` callback, only the optimizer calls the schedule.
other = Net()
other.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=Untracked()))
other.fit(x, y, epochs=1, verbose=0)
