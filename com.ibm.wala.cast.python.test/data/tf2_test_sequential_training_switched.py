# Test https://github.com/wala/ML/issues/1249: a `Sequential` called with `training=True` twice gives
# its layer `True` alone, and one called with `False` and then `True` gives its layer both.
import tensorflow as tf

SEEN = []


class TrainedGate(tf.keras.layers.Layer):

    def call(self, x, training=None):
        SEEN.append(("TrainedGate", training))
        return x


class SwitchedGate(tf.keras.layers.Layer):

    def call(self, x, training=None):
        SEEN.append(("SwitchedGate", training))
        return x


xs = tf.ones([4, 3])

trained = tf.keras.Sequential([TrainedGate()])
trained(xs, training=True)
trained(xs, training=True)

switched = tf.keras.Sequential([SwitchedGate()])
switched(xs, training=False)
switched(xs, training=True)

assert set(SEEN) == {
    ("TrainedGate", True),
    ("SwitchedGate", False),
    ("SwitchedGate", True),
}, SEEN
