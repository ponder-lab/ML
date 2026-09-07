# The Sequential counterpart of tf2_test_model_attributes2.py: the same two-layer network, the
# same weights, built from a layer LIST rather than from `inputs`/`outputs`. wala/ML#832's weights
# half is that the weight machinery anchors on the `outputs` constructor argument, which a
# Sequential frame does not have, so weight-attribute reads on a Sequential-built model resolve
# nothing even though the layer list determines every shape below.
import tensorflow as tf


def f(a):
    pass


model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(3,)),
        tf.keras.layers.Dense(5, activation=tf.nn.softmax),
    ]
)

for i in model.trainable_weights:
    assert i.dtype == tf.float32
    assert i.shape in [(3, 4), (4,), (4, 5), (5,)]
    f(i)
