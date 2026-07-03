# Pins wala/ML#670's fixes directly: `GlobalAveragePooling1D` is modeled (rank-3 input, temporal
# axis dropped), so the functional model's weight walk resolves the downstream `Dense` kernel and
# bias concretely.
import tensorflow as tf


def consume(t):
    pass


a = tf.keras.Input(shape=(10, 8))
b = tf.keras.layers.GlobalAveragePooling1D()(a)
c = tf.keras.layers.Dense(5)(b)
model = tf.keras.Model(a, c)

for w in model.trainable_weights:
    consume(w)

assert tuple(model.trainable_weights[0].shape) == (8, 5)
assert tuple(model.trainable_weights[1].shape) == (5,)
assert all(w.dtype == tf.float32 for w in model.trainable_weights)
