# Test: a tower of ten wrapper layers, each building its inner layer lazily on its first call
# and used through fit. Past the receiver-context depth cap the innermost layer's build still
# dispatches: the degraded trampoline is keyed on the receiver alone, not on its caller's.
import numpy as np
import tensorflow as tf


class Leaf(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.w = self.add_weight(shape=[input_shape[-1]], initializer="ones", name="w")

    def call(self, inputs):
        return inputs * self.w


class Wrap10(tf.keras.layers.Layer):
    def __init__(self, inner=None, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner

    def build(self, input_shape):
        self.inner.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        return self.inner(inputs) + 1.0


class Wrap9(tf.keras.layers.Layer):
    def __init__(self, inner=None, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner

    def build(self, input_shape):
        self.inner.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        return self.inner(inputs) + 1.0


class Wrap8(tf.keras.layers.Layer):
    def __init__(self, inner=None, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner

    def build(self, input_shape):
        self.inner.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        return self.inner(inputs) + 1.0


class Wrap7(tf.keras.layers.Layer):
    def __init__(self, inner=None, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner

    def build(self, input_shape):
        self.inner.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        return self.inner(inputs) + 1.0


class Wrap6(tf.keras.layers.Layer):
    def __init__(self, inner=None, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner

    def build(self, input_shape):
        self.inner.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        return self.inner(inputs) + 1.0


class Wrap5(tf.keras.layers.Layer):
    def __init__(self, inner=None, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner

    def build(self, input_shape):
        self.inner.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        return self.inner(inputs) + 1.0


class Wrap4(tf.keras.layers.Layer):
    def __init__(self, inner=None, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner

    def build(self, input_shape):
        self.inner.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        return self.inner(inputs) + 1.0


class Wrap3(tf.keras.layers.Layer):
    def __init__(self, inner=None, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner

    def build(self, input_shape):
        self.inner.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        return self.inner(inputs) + 1.0


class Wrap2(tf.keras.layers.Layer):
    def __init__(self, inner=None, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner

    def build(self, input_shape):
        self.inner.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        return self.inner(inputs) + 1.0


class Wrap1(tf.keras.layers.Layer):
    def __init__(self, inner=None, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner

    def build(self, input_shape):
        self.inner.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        return self.inner(inputs) + 1.0


class Model(tf.keras.Model):
    def __init__(self, tower=None, **kwargs):
        super().__init__(**kwargs)
        self.tower = tower

    def call(self, inputs):
        return tf.reduce_sum(self.tower(inputs), axis=-1)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(self(x, training=True) - y))
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}


x = np.ones((8, 4), dtype=np.float32)
y = np.ones((8,), dtype=np.float32)
model = Model(
    tower=Wrap1(
        inner=Wrap2(
            inner=Wrap3(
                inner=Wrap4(
                    inner=Wrap5(
                        inner=Wrap6(
                            inner=Wrap7(
                                inner=Wrap8(inner=Wrap9(inner=Wrap10(inner=Leaf())))
                            )
                        )
                    )
                )
            )
        )
    )
)
model.compile(optimizer="sgd", loss="mse")
model.fit(x, y, epochs=1, verbose=0)
assert (
    model.tower.inner.inner.inner.inner.inner.inner.inner.inner.inner.inner.w.shape
    == (4,)
)
