# Test for wala/ML#832: a `Sequential` model whose list holds a layer other than `Dense` has weights
# its list does not give, here the convolution's kernel and bias, so its weights are tensors of
# shapes the analysis does not know, not the `Dense` layer's alone.
import tensorflow as tf


def f(a):
    pass


model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv1D(4, 3),
        tf.keras.layers.Dense(5),
    ]
)
model(tf.ones((2, 8, 3)))

for i in model.trainable_weights:
    assert i.dtype == tf.float32
    assert i.shape in [(3, 3, 4), (4,), (4, 5), (5,)]
    f(i)
