# Test https://github.com/wala/ML/issues/996: the other forms a weight constraint takes, every
# `add_weight` argument positional, and a function or an object constraint on `tf.Variable`.
import tensorflow as tf


def norm_positional(w):
    assert w.shape == (4, 3)
    return tf.nn.relu(w)


def norm_variable(w):
    assert w.shape == (6, 1)
    return tf.nn.relu(w)


class ObjNorm(tf.keras.constraints.Constraint):
    def __call__(self, w):
        assert w.shape == (7, 1)
        return tf.nn.relu(w)


class PosLayer(tf.keras.layers.Layer):
    # Every add_weight argument positional: no keyword binding on the way.
    def build(self, input_shape):
        self.w = self.add_weight(
            "w", (4, 3), "float32", "ones", None, True, norm_positional
        )

    def call(self, x):
        return tf.matmul(x, self.w)


# A function constraint on tf.Variable directly.
v = tf.Variable(tf.ones((6, 1)), constraint=norm_variable)
# An object constraint on tf.Variable directly.
u = tf.Variable(tf.ones((7, 1)), constraint=ObjNorm())

layer = PosLayer()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
with tf.GradientTape() as tape:
    loss = tf.reduce_sum(layer(tf.ones((2, 4)))) + tf.reduce_sum(v) + tf.reduce_sum(u)
variables = layer.trainable_variables + [v, u]
gradients = tape.gradient(loss, variables)
optimizer.apply_gradients(zip(gradients, variables))
