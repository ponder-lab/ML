# Test https://github.com/wala/ML/issues/996: a model rebuilt from its configuration keeps its
# layers' constraints, so training the rebuilt model applies the original constraint to the weight.
# The configuration round trip goes through `serialize_keras_object`, `layers.deserialize`,
# `constraints.serialize`, and `constraints.get`, as a nested layer does it.
import numpy as np
import tensorflow as tf


def consume_norm(w):
    assert w.shape == (4, 3) and w.dtype == tf.float32
    return w


class Norm(tf.keras.constraints.Constraint):
    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, w):
        consume_norm(w)
        return self.scale * tf.nn.relu(w)

    def get_config(self):
        return {"scale": self.scale}


class Mink(tf.keras.layers.Layer):
    def __init__(self, w_constraint=None, **kwargs):
        super().__init__(**kwargs)
        if w_constraint is None:
            w_constraint = tf.keras.constraints.NonNeg()
        self.w_constraint = tf.keras.constraints.get(w_constraint)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(4, 3), initializer="ones", name="w", constraint=self.w_constraint
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"w_constraint": tf.keras.constraints.serialize(self.w_constraint)}
        )
        return config


class Outer(tf.keras.layers.Layer):
    def __init__(self, distance=None, **kwargs):
        super().__init__(**kwargs)
        self.distance = distance

    def build(self, input_shape):
        self.distance.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        return self.distance(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"distance": tf.keras.utils.serialize_keras_object(self.distance)}
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["distance"] = tf.keras.layers.deserialize(
            config["distance"], custom_objects={"Mink": Mink, "Norm": Norm}
        )
        return cls(**config)


class Net(tf.keras.Model):
    def __init__(self, kernel=None, **kwargs):
        super().__init__(**kwargs)
        self.kernel = kernel

    def call(self, inputs):
        return self.kernel(inputs)

    def get_config(self):
        return {"kernel": tf.keras.utils.serialize_keras_object(self.kernel)}

    @classmethod
    def from_config(cls, config):
        config["kernel"] = tf.keras.layers.deserialize(
            config["kernel"],
            custom_objects={"Outer": Outer, "Mink": Mink, "Norm": Norm},
        )
        return cls(**config)


def restart(model):
    return model.from_config(model.get_config())


x = np.ones((8, 4), dtype=np.float32)
y = np.ones((8, 3), dtype=np.float32)
original = Net(kernel=Outer(distance=Mink(w_constraint=Norm(scale=2.0))))
rebuilt = restart(original)
rebuilt.compile(optimizer="sgd", loss="mse")
rebuilt.fit(x, y, epochs=1, verbose=0)
