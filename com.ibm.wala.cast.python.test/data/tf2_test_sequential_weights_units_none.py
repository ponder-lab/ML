# A Sequential model whose Dense layer's `units` may, as far as the analysis knows, be `None`: the
# value comes from a function that returns its parameter, whose default is `None`, or a literal.
# At run time it is 4, but the analysis cannot rule `None` out, so the model's weights are tensors
# of unknown shape rather than a crash.
import tensorflow as tf


def f(a):
    pass


def width(n=None):
    return 4 if n is None else n


model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(width(), activation=tf.nn.relu, input_shape=(3,)),
        tf.keras.layers.Dense(5, activation=tf.nn.softmax),
    ]
)

for i in model.trainable_weights:
    assert i.dtype == tf.float32
    assert i.shape in [(3, 4), (4,), (4, 5), (5,)]
    f(i)
