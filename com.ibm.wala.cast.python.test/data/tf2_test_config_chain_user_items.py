# Test https://github.com/wala/ML/issues/997: the layer configurations are a dict on one path and a
# user object with its own `items` method on the other, so the loop over `items` must read the
# dict's fields and dispatch the user's method.
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


def consume_registry(marker):
    return marker


class Registry:
    def __init__(self, kernel_config):
        self.kernel_config = kernel_config
        self.marker = 1

    def items(self):
        consume_registry(self.marker)
        return [("kernel", self.kernel_config)]


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
    def from_config(cls, config, registry=False):
        layer_configs = config["layers"]
        if registry:
            layer_configs = Registry(layer_configs["kernel"])
        built_layers = {}
        for layer_name, layer_config in layer_configs.items():
            built_layers[layer_name] = tf.keras.layers.deserialize(
                layer_config, custom_objects=CUSTOM
            )
        return cls(kernel=built_layers["kernel"])


def restart(model, registry):
    return model.from_config(model.get_config(), registry=registry)


x = np.ones((8, 4), dtype=np.float32)
y = np.ones((8, 3), dtype=np.float32)
for registry in (False, True):
    original = Net(kernel=Mink(w_constraint=Norm(scale=2.0)))
    rebuilt = restart(original, registry)
    rebuilt.compile(optimizer="sgd", loss="mse")
    rebuilt.fit(x, y, epochs=1, verbose=0)
