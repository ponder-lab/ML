import tensorflow as tf


def reached_epoch_end(logs):
    return logs


class Recorder(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        reached_epoch_end(logs)


inputs = tf.keras.Input(shape=(4,))
outputs = tf.keras.layers.Dense(1)(inputs)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="sgd", loss="mse")
callbacks = []
callbacks.append(Recorder())
model.fit(tf.ones([8, 4]), tf.ones([8, 1]), epochs=1, verbose=0, callbacks=callbacks)
