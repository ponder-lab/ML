# Test https://github.com/wala/ML/issues/1328: a function the call graph does not reach may still
# run, but its `update`, `setdefault` or `add` on a value it cannot trace changes no list, since no
# list has those methods. So the slice `inputs[0:-1]` holds the data inputs alone, not the group
# tensor it drops.
import tensorflow as tf


def consume(t):
    assert t.shape == (2, 3)
    assert t.dtype == tf.float32


class Gate(tf.keras.layers.Layer):
    def call(self, inputs):
        data = inputs[0:-1]
        consume(data[0])
        return data[0]

    def get_config(self):
        config = super().get_config()
        config.update({"rate": 0.5})
        config.setdefault("axis", -1)
        return config


def collect(make):
    seen = make()
    seen.add(0)
    return seen


x = tf.ones([2, 3])
groups = tf.zeros([2, 1], dtype=tf.int32)
Gate()([x, x, groups])
