# B half of the wala/ML#678 repro: this script and its sibling each define a Keras subclass
# named `GenGPT2` whose `__init__` references the class by name in `super(GenGPT2, self)`; both
# classes' methods must keep their call-graph nodes.
import tensorflow as tf


class GenGPT2(tf.keras.Model):
    def __init__(self, param, **kwargs):
        super(GenGPT2, self).__init__(**kwargs)
        self.param = param

    def call(self, inputs):
        return inputs

    def predict(self, inputs):
        return self(inputs)


m = GenGPT2(param=4)
r = m.predict(tf.ones((3, 3)))
assert r.shape == (3, 3)
