# Test: a model whose kernel is a gate over scaled, distance-based subnets wrapping a constrained
# distance layer, built lazily on the first call at every level, used through fit, evaluate and
# predict, then rebuilt from its config and used again. Every layer method body runs under a
# bounded number of contexts: one per receiver chain and call site, not one per call path.
import numpy as np
import tensorflow as tf


def consume_norm(w):
    return w


class Norm(tf.keras.constraints.Constraint):
    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, w):
        consume_norm(w)
        return self.scale * tf.nn.relu(w)

    def get_config(self):
        return {"scale": self.scale}


class Minkowski(tf.keras.layers.Layer):
    def __init__(self, w_constraint=None, **kwargs):
        super().__init__(**kwargs)
        if w_constraint is None:
            w_constraint = tf.keras.constraints.NonNeg()
        self.w_constraint = tf.keras.constraints.get(w_constraint)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=[input_shape[0][-1]],
            initializer="ones",
            name="w",
            constraint=self.w_constraint,
        )

    def call(self, inputs):
        z_q, z_r = inputs
        return tf.reduce_sum(tf.pow(tf.abs(z_q - z_r), 2.0) * self.w, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {"w_constraint": tf.keras.constraints.serialize(self.w_constraint)}
        )
        return config


class DistanceBased(tf.keras.layers.Layer):
    def __init__(self, distance=None, **kwargs):
        super().__init__(**kwargs)
        self.distance = distance

    def build(self, input_shape):
        self.distance.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        return tf.exp(-self.distance(inputs))

    def get_config(self):
        config = super().get_config()
        config.update(
            {"distance": tf.keras.utils.serialize_keras_object(self.distance)}
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["distance"] = tf.keras.layers.deserialize(
            config["distance"], custom_objects={"Minkowski": Minkowski, "Norm": Norm}
        )
        return cls(**config)


class GateMulti(tf.keras.layers.Layer):
    def __init__(self, subnets=None, **kwargs):
        super().__init__(**kwargs)
        self.subnets = subnets

    def build(self, inputs_shape):
        input_shape_less_group = inputs_shape[0:-1]
        for subnet in self.subnets:
            subnet.build(input_shape_less_group)
        super().build(inputs_shape)

    def call(self, inputs):
        z_q, z_r, groups = inputs
        out = self.subnets[0]([z_q, z_r])
        for subnet in self.subnets[1:]:
            out = out + subnet([z_q, z_r])
        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "subnets": [
                    tf.keras.utils.serialize_keras_object(s) for s in self.subnets
                ]
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["subnets"] = [
            tf.keras.layers.deserialize(
                s,
                custom_objects={
                    "Scaled": Scaled,
                    "DistanceBased": DistanceBased,
                    "Minkowski": Minkowski,
                    "Norm": Norm,
                },
            )
            for s in config["subnets"]
        ]
        return cls(**config)


class Base(tf.keras.Model):
    def __init__(self, stimuli=None, kernel=None, **kwargs):
        super().__init__(**kwargs)
        self.stimuli = stimuli
        self.kernel = kernel

    def get_config(self):
        layer_configs = {
            "stimuli": tf.keras.utils.serialize_keras_object(self.stimuli),
            "kernel": tf.keras.utils.serialize_keras_object(self.kernel),
        }
        return {"layers": layer_configs}

    @classmethod
    def from_config(cls, config):
        built = {}
        for name, layer_config in config["layers"].items():
            built[name] = tf.keras.layers.deserialize(
                layer_config,
                custom_objects={
                    "GateMulti": GateMulti,
                    "DistanceBased": DistanceBased,
                    "Minkowski": Minkowski,
                    "Norm": Norm,
                },
            )
        return cls(**built)


class Rank(Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        stimulus_set, groups = inputs
        z = self.stimuli(stimulus_set)
        z_q, z_r = z[:, 0], z[:, 1]
        sim = self.kernel([z_q, z_r, groups])
        return sim

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(self(x, training=True) - y))
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {"loss": loss}


class Scaled(tf.keras.layers.Layer):
    def __init__(self, inner=None, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.inner = inner
        self.scale = scale

    def build(self, input_shape):
        self.inner.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        return self.inner(inputs) * self.scale

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "inner": tf.keras.utils.serialize_keras_object(self.inner),
                "scale": self.scale,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["inner"] = tf.keras.layers.deserialize(
            config["inner"],
            custom_objects={
                "DistanceBased": DistanceBased,
                "Minkowski": Minkowski,
                "Norm": Norm,
            },
        )
        return cls(**config)


def build_model():
    subnets = [
        Scaled(inner=DistanceBased(distance=Minkowski(w_constraint=Norm(2.0))))
        for _ in range(3)
    ]
    return Rank(
        stimuli=tf.keras.layers.Embedding(10, 4), kernel=GateMulti(subnets=subnets)
    )


x = (np.ones((8, 2), dtype=np.int32), np.zeros((8,), dtype=np.int32))
y = np.ones((8,), dtype=np.float32)
model = build_model()
model.compile(optimizer="sgd", loss="mse")
model.fit(x, y, epochs=1, verbose=0)
model.evaluate(x, y, verbose=0)
model.predict(x, verbose=0)
rebuilt = model.from_config(model.get_config())
rebuilt.compile(optimizer="sgd", loss="mse")
rebuilt.fit(x, y, epochs=1, verbose=0)
rebuilt.evaluate(x, y, verbose=0)
