"""Regression guard: layer-output → user-defined `Model.__call__` → script-level consumer.

Variant of ``tf2_test_layer_output_param.py`` that interposes a user-defined
``tf.keras.Model`` subclass between the ``Dense`` layers and the script-level consumer.
Same recovery mechanism as the companion fixture (``DenseCall.getDefaultShapes`` SSA-chain
fallback), but exercised across one extra level of call indirection.
"""

import tensorflow as tf


def consume(tensor):
    pass


class NeuralNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(16, activation="relu")
        self.d2 = tf.keras.layers.Dense(10)

    def __call__(self, x):
        x = self.d1(x)
        return self.d2(x)


batch_x = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
neural_net = NeuralNet()
pred = neural_net(batch_x)
consume(pred)
