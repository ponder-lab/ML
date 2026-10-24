# Test `TensorFlowApis`: each function makes one call of interest, and the test reads the APIs it
# invokes.
import json

import numpy as np
import scipy.sparse as sp
import scipy.special
from scipy import special

import tensorflow as tf
import tensorflow as T
from tensorflow import reshape as rs


def direct_reshape(x):
    return tf.reshape(x, [6])


def direct_shape(x):
    return tf.shape(x)


def direct_zeros():
    return tf.zeros([2])


def direct_random_normal():
    return tf.random.normal([2])


def aliased_reshape(x):
    return rs(x, [6])


def method_set_shape(t):
    t.set_shape([2])
    return t


def keras_backend():
    return tf.keras.backend.floatx()


def either(c):
    f = tf.zeros if c else tf.ones
    return f([2])


def helper(x):
    return tf.zeros([2])


def user_function(x):
    return helper(x)


def unresolved(f):
    return f("1")


def module_alias(x):
    return T.shape(x)


def rebound_local(x):
    s = tf.shape
    return s(x)


S = tf.shape


def rebound_global(x):
    return S(x)


def image_resize(img):
    return tf.image.resize(img, [2, 2])


def dense_call(layer, x):
    return layer(x)


def dropout_call(layer, x):
    return layer(x)


tf.myfn = helper


def stored_on_module(x):
    return tf.myfn(x)


def aliased_sigmoid(x):
    return tf.math.sigmoid(x)


def apply_application(model, x):
    return model(x)


def serialize_layer(layer):
    return tf.keras.layers.serialize(layer)


class Weighted(tf.keras.Model):
    def build(self, input_shape):
        self.w = self.add_weight(name="w", shape=(1,))

    def call(self, inputs):
        return inputs * self.w


def direct_add_weight(m):
    return tf.keras.Model.add_weight(m, name="v", shape=(1,))


def slice_list(values):
    return values[1:3]


def slice_builtin():
    return slice(1, 3)


def length(values):
    return len(values)


def np_asarray(a):
    return np.asarray(a)


def np_array(a):
    return np.array(a)


def scipy_expit(a):
    return scipy.special.expit(a)


def special_expit(a):
    return special.expit(a)


def sparse_diags(a):
    return sp.diags(a)


def scipy_root_diags(a):
    import scipy

    return scipy.sparse.diags(a)


def scipy_root_expit(a):
    import scipy

    return scipy.special.expit(a)


def to_bool(t):
    return bool(t)


def run_rnn(x):
    layer = tf.keras.layers.RNN(
        tf.keras.layers.SimpleRNNCell(4), return_state=True, return_sequences=True
    )
    return layer(x)[0]


x = tf.ones([2, 3])
assert direct_reshape(x).shape == (6,)
assert direct_shape(x).numpy().tolist() == [2, 3]
assert direct_zeros().shape == (2,)
assert direct_random_normal().shape == (2,)
assert aliased_reshape(x).shape == (6,)
# Only a parsed feature's tensor carries a modeled `set_shape`.
example = tf.train.Example(
    features=tf.train.Features(
        feature={"x": tf.train.Feature(float_list=tf.train.FloatList(value=[1.0, 2.0]))}
    )
)
parsed = tf.io.parse_single_example(
    example.SerializeToString(), {"x": tf.io.FixedLenFeature([2], tf.float32)}
)
assert method_set_shape(parsed["x"]).shape == (2,)
assert keras_backend() == "float32"
assert either(True).shape == (2,) and either(False).shape == (2,)
assert user_function(x).shape == (2,)
assert unresolved(json.loads) == 1
assert module_alias(x).numpy().tolist() == [2, 3]
assert rebound_local(x).numpy().tolist() == [2, 3]
assert rebound_global(x).numpy().tolist() == [2, 3]
assert image_resize(tf.ones([4, 4, 3])).shape == (2, 2, 3)
assert dense_call(tf.keras.layers.Dense(4), x).shape == (2, 4)
assert dropout_call(tf.keras.layers.Dropout(0.5), x).shape == (2, 3)
assert stored_on_module(x).shape == (2,)
assert aliased_sigmoid(x).shape == (2, 3)
assert apply_application(
    tf.keras.applications.MobileNetV2(weights=None, input_shape=(32, 32, 3)),
    tf.ones([1, 32, 32, 3]),
).shape == (1, 1000)
assert serialize_layer(tf.keras.layers.Dense(4))["class_name"] == "Dense"
weighted = Weighted()
weighted.build((1,))
assert direct_add_weight(weighted).shape == (1,)
assert run_rnn(tf.ones([2, 3, 5])).shape == (2, 3, 4)
assert slice_list([1, 2, 3, 4]) == [2, 3]
assert slice_builtin() == slice(1, 3)
assert length([1, 2]) == 2
assert np_asarray([1.0, 2.0]).shape == (2,)
assert np_array([1.0, 2.0]).shape == (2,)
assert scipy_expit(0.0) == 0.5
assert special_expit(0.0) == 0.5
assert sparse_diags([1.0, 2.0]).shape == (2, 2)
assert scipy_root_diags([1.0, 2.0]).shape == (2, 2)
assert scipy_root_expit(0.0) == 0.5
assert to_bool(tf.constant(True)) is True
