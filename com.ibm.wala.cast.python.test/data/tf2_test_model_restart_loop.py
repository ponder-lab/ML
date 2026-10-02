# Test https://github.com/wala/ML/issues/210: a model rebuilt from its own config in a loop, the
# rebuilt model stored back where the next round reads it, must not nest a receiver context per
# round.
import tensorflow as tf


class Net(tf.keras.Model):
    def __init__(self, units=2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = tf.keras.layers.Dense(units)

    def call(self, inputs):
        return self.dense(inputs)

    def get_config(self):
        return {"units": self.units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _new_model(model):
    return model.from_config(model.get_config())


def consume(y):
    assert y.shape == (3, 2)
    return y


class Restarter:
    def __init__(self, model, n_restart):
        self.model = model
        self.n_restart = n_restart

    def fit(self, x):
        for _ in range(self.n_restart):
            restarted = _new_model(self.model)
            consume(restarted(x))
            self.model = restarted
        return self.model


Restarter(Net(), 3).fit(tf.ones((3, 4)))
