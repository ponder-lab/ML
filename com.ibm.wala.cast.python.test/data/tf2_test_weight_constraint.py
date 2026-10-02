# Test https://github.com/wala/ML/issues/996: a weight constraint passed to `add_weight` is applied
# to the weight by the optimizer, so the constraint's `__call__` receives the weight.
import tensorflow as tf


class NonNegNorm(tf.keras.constraints.Constraint):
    def __init__(self, scale=1.0, p=2.0, axis=0):
        self.scale = scale
        self.p = p
        self.axis = axis

    def __call__(self, w):
        # Applied by the optimizer to the stored-attribute weight below.
        assert w.shape == (4, 3) and w.dtype == tf.float32
        w = w * tf.cast(tf.math.greater_equal(w, 0.0), tf.float32)
        return self.scale * (
            w
            / (
                1e-7
                + tf.pow(
                    tf.reduce_sum(w**self.p, axis=self.axis, keepdims=True),
                    tf.divide(1.0, self.p),
                )
            )
        )


class InlineNorm(tf.keras.constraints.Constraint):
    def __call__(self, w):
        # Applied by the optimizer to the inline-constrained weight below.
        assert w.shape == (5, 2) and w.dtype == tf.float32
        return tf.nn.relu(w)


class Stored(tf.keras.layers.Layer):
    # The constraint arrives through a constructor keyword, is stored, and is passed to
    # `add_weight` in `build`.
    def __init__(self, w_constraint=None, **kwargs):
        super().__init__(**kwargs)
        if w_constraint is None:
            w_constraint = tf.keras.constraints.NonNeg()
        self.w_constraint = tf.keras.constraints.get(w_constraint)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(4, 3),
            initializer="ones",
            trainable=True,
            name="w",
            constraint=self.w_constraint,
        )

    def call(self, x):
        return tf.matmul(x, self.w)


class Inline(tf.keras.layers.Layer):
    # The constraint is constructed inline at `add_weight`.
    def build(self, input_shape):
        self.k = self.add_weight(
            shape=(5, 2), initializer="ones", name="k", constraint=InlineNorm()
        )

    def call(self, x):
        return tf.matmul(x, self.k)


def consume_stored(y):
    assert y.shape == (2, 3)
    return y


def consume_inline(y):
    assert y.shape == (2, 2)
    return y


stored = Stored(w_constraint=NonNegNorm(scale=1.0, p=2.0, axis=0))
inline = Inline()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
x = tf.ones((2, 4))
z = tf.ones((2, 5))
for _ in range(2):
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(consume_stored(stored(x))) + tf.reduce_sum(
            consume_inline(inline(z))
        )
    variables = stored.trainable_variables + inline.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
