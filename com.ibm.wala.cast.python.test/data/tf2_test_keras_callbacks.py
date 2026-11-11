import tensorflow as tf


def reached_epoch_end(logs):
    return logs


def reached_train_batch_end(logs):
    return logs


def reached_test_end(logs):
    return logs


def reached_predict_end(logs):
    return logs


def model_output(t):
    assert t.shape == (2, 1)
    assert t.dtype == tf.float32
    return t


def later_epoch_output(t):
    assert t.shape == (2, 1)
    assert t.dtype == tf.float32
    return t


class Recorder(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        if epoch > 0:
            later_epoch_output(self.model(tf.ones([2, 4])))

    def on_epoch_end(self, epoch, logs=None):
        reached_epoch_end(logs)
        model_output(self.model(tf.ones([2, 4])))

    def on_train_batch_end(self, batch, logs=None):
        reached_train_batch_end(logs)

    def on_test_end(self, logs=None):
        reached_test_end(logs)

    def on_predict_end(self, logs=None):
        reached_predict_end(logs)


inputs = tf.keras.Input(shape=(4,))
outputs = tf.keras.layers.Dense(1)(inputs)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="sgd", loss="mse")
x = tf.ones([8, 4])
y = tf.ones([8, 1])
recorder = Recorder()
model.fit(x, y, epochs=2, verbose=0, callbacks=[recorder])
model.evaluate(x, y, verbose=0, callbacks=[recorder])
model.predict(x, verbose=0, callbacks=[recorder])
