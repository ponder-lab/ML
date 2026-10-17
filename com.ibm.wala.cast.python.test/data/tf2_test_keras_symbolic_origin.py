# Test for wala/ML#1101: a Keras symbolic tensor, from `tf.keras.layers.Input` through the layer
# applications and operations applied to it, and into the parameters of the functions it is
# passed to, is told apart from an eager tensor of the same type.
import tensorflow as tf


def backbone(input_data):
    conv = tf.keras.layers.Conv2D(8, 3, padding="same")(input_data)
    return tf.image.resize(
        conv, (conv.shape[1] * 2, conv.shape[2] * 2), method="nearest"
    )


def model_fn(input_layer):
    return backbone(input_layer)


def decode(conv_output):
    return conv_output * 2.0


def eager_only(x):
    return x * 2.0


outputs = []
for i, out in enumerate([model_fn(tf.keras.layers.Input([16, 16, 3]))]):
    outputs.append(decode(out))
assert type(outputs[0]).__name__ == "KerasTensor"

eager = model_fn(tf.ones([1, 16, 16, 3]))
assert isinstance(eager, tf.Tensor)

result = eager_only(tf.ones([2]))
assert isinstance(result, tf.Tensor)
