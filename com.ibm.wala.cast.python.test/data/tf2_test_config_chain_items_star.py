# Test https://github.com/wala/ML/issues/997: a model-level configuration round trip, the layers rebuilt in a loop over items and bound by a dict unpacking.
import copy
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


CUSTOM = {"Mink": Mink, "Norm": Norm}


class Net(tf.keras.Model):
    def __init__(self, kernel=None, **kwargs):
        super().__init__(**kwargs)
        self.kernel = kernel

    def call(self, inputs):
        return self.kernel(inputs)

    def get_config(self):
        return {
            "layers": {"kernel": tf.keras.utils.serialize_keras_object(self.kernel)}
        }

    @classmethod
    def from_config(cls, config):
        built_layers = {}
        for layer_name, layer_config in config["layers"].items():
            built_layers[layer_name] = tf.keras.layers.deserialize(
                layer_config, custom_objects=CUSTOM
            )
        return cls(**built_layers)


def restart(model):
    return model.from_config(model.get_config())


x = np.ones((8, 4), dtype=np.float32)
y = np.ones((8, 3), dtype=np.float32)
original = Net(kernel=Mink(w_constraint=Norm(scale=2.0)))
rebuilt = restart(original)
rebuilt.compile(optimizer="sgd", loss="mse")
rebuilt.fit(x, y, epochs=1, verbose=0)
