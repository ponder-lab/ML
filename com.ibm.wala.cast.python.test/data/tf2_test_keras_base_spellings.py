# Test for wala/ML#1127: a program class whose base is a Keras layer or model spelled through the
# standalone `keras` package, TensorFlow's legacy `tensorflow.python.keras`, or `tf.compat`, or
# whose base is a built-in layer, is a Keras layer or model.
import keras
import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import Layer as BaseLayer
from tensorflow.python.keras.engine.training import Model as TrainingModel
from tensorflow.python.keras.layers import Layer as LegacyLayer


class KerasLayer(keras.layers.Layer):
    pass


class KerasModel(keras.Model):
    pass


class KerasDense(keras.layers.Dense):
    pass


class LegacyLayerSub(LegacyLayer):
    pass


class BaseLayerSub(BaseLayer):
    pass


class TrainingModelSub(TrainingModel):
    pass


class CompatV1Layer(tf.compat.v1.keras.layers.Layer):
    pass


class CompatV1Model(tf.compat.v1.keras.Model):
    pass


class CompatV2Layer(tf.compat.v2.keras.layers.Layer):
    pass


class MaxPooling2DSub(tf.keras.layers.MaxPooling2D):
    pass


class Conv3DSub(tf.keras.layers.Conv3D):
    pass


class MultiHeadAttentionSub(tf.keras.layers.MultiHeadAttention):
    pass


class Layer:
    pass


class NotKeras(Layer):
    pass


for cls in (
    KerasLayer,
    KerasModel,
    KerasDense,
    CompatV1Layer,
    CompatV1Model,
    CompatV2Layer,
    MaxPooling2DSub,
    Conv3DSub,
    MultiHeadAttentionSub,
):
    assert issubclass(cls, tf.keras.layers.Layer)

for cls in (LegacyLayerSub, BaseLayerSub, TrainingModelSub):
    assert issubclass(cls, BaseLayer) and not issubclass(cls, tf.keras.layers.Layer)

assert issubclass(MaxPooling2DSub, tf.keras.layers.MaxPool2D)
assert not issubclass(NotKeras, tf.keras.layers.Layer)
