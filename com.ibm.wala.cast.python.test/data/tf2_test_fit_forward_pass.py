# Test https://github.com/wala/ML/issues/997: a model trained only through `fit` runs its
# forward pass, through its own `train_step` when it overrides one, so its layers' `call`, their
# lazily built weights, and the weights' constraints are reached.
import numpy as np
import tensorflow as tf


def consume_inner(x):
    # Traced by `fit` with the batch axis unspecified.
    assert x.shape[1:] == (4,) and x.dtype == tf.float32
    return x


def consume_step_x(x):
    assert x.shape[1:] == (4,) and x.dtype == tf.float32
    return x


def consume_step_y(y):
    assert y.shape[1:] == (3,) and y.dtype == tf.float32
    return y


def consume_prediction(p):
    assert p.shape == (8, 3) and p.dtype == tf.float32
    return p


class FitNorm(tf.keras.constraints.Constraint):
    def __call__(self, w):
        # Applied by the optimizer after each update inside `fit`.
        assert w.shape == (4, 3) and w.dtype == tf.float32
        return tf.nn.relu(w)


class Inner(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.w = self.add_weight(shape=(4, 3), initializer="ones", constraint=FitNorm())

    def call(self, x):
        consume_inner(x)
        return tf.matmul(x, self.w)


class Plain(tf.keras.Model):
    # No `train_step` override: the default one runs the forward pass.
    def __init__(self):
        super().__init__()
        self.inner = Inner()

    def call(self, inputs):
        return self.inner(inputs)


class Stepped(tf.keras.Model):
    # A `train_step` override, reached by `fit` and given the `(x, y)` batch.
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(3)

    def call(self, inputs):
        return self.dense(inputs)

    def train_step(self, data):
        x, y = data
        consume_step_x(x)
        consume_step_y(y)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(self(x, training=True) - y))
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}


x = np.ones((8, 4), dtype=np.float32)
y = np.ones((8, 3), dtype=np.float32)

plain = Plain()
plain.compile(optimizer="sgd", loss="mse")
plain.fit(x, y, epochs=1, verbose=0)
consume_prediction(plain.predict(x, verbose=0))

stepped = Stepped()
stepped.compile(optimizer="sgd", loss="mse")
stepped.fit(x, y, epochs=1, verbose=0)
