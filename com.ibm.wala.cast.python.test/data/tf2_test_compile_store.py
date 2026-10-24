# Test https://github.com/wala/ML/issues/1146: `compile` stores the loss and the optimizer on the
# model, so a `train_step` that calls `self.loss` and `self.optimizer` reaches them.
import numpy as np
import tensorflow as tf

seen = []


class MyLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        assert y_pred.shape[-1] == 3 and y_pred.dtype == tf.float32
        seen.append("loss")
        return tf.reduce_mean(tf.square(y_true - y_pred))


class Schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __call__(self, step):
        assert step.shape == ()
        seen.append("schedule")
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
            loss = self.loss(y, self(x, training=True))
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}


class BatchLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        assert y_pred.shape == (4, 3) and y_pred.dtype == tf.float32
        seen.append("batch loss")
        return tf.reduce_mean(tf.square(y_true - y_pred))


class BatchNet(tf.keras.Model):
    # Overrides `train_on_batch`, not `train_step`: the loss is called from a private step.
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(3)

    def call(self, inputs):
        return self.dense(inputs)

    def __train_step(self, x, y):
        with tf.GradientTape() as tape:
            loss = self.loss(y, self(x, training=True))
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    def train_on_batch(self, x, y):
        return self.__train_step(x, y)


net = Net()
net.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=Schedule()), loss=MyLoss())
net.fit(
    np.ones((4, 2), dtype=np.float32),
    np.ones((4, 3), dtype=np.float32),
    epochs=1,
    verbose=0,
)
batch = BatchNet()
batch.compile(optimizer="sgd", loss=BatchLoss())
batch.train_on_batch(tf.ones((4, 2)), tf.ones((4, 3)))
assert "loss" in seen and "schedule" in seen and "batch loss" in seen
