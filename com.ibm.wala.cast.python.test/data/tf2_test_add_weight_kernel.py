# Test a weight that a Keras model creates with `self.add_weight(...)` and reads in `call`.
# A model is a layer at run time, so its `add_weight` creates a weight as a layer's does: the
# weight has the `shape` argument's shape and the `dtype` argument's dtype, float32 when none is
# given. Each model passes the weight to its own sink, and the product of a float input with the
# weight to one or two more, through `tf.matmul` and `@`.

import tensorflow as tf


def kernel_build(k):
    assert k.shape == (4, 2)
    assert k.dtype == tf.float32


def matmul_build(y):
    assert y.shape == (5, 2)
    assert y.dtype == tf.float32


def at_build(y):
    assert y.shape == (5, 2)
    assert y.dtype == tf.float32


class BuildModel(tf.keras.Model):
    """Creates its weight in `build`, from the input's last axis, with no `dtype`."""

    def __init__(self):
        super().__init__()
        self.kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel", shape=[input_shape[-1], 2], initializer="glorot_uniform"
        )

    def call(self, inputs):
        kernel_build(self.kernel)
        matmul_build(tf.matmul(inputs, self.kernel))
        at_build(inputs @ self.kernel)
        return inputs


def kernel_init(k):
    assert k.shape == (4, 2)
    assert k.dtype == tf.float32


def at_init(y):
    assert y.shape == (5, 2)
    assert y.dtype == tf.float32


class InitModel(tf.keras.Model):
    """Creates its weight in `__init__`."""

    def __init__(self):
        super().__init__()
        self.kernel = self.add_weight(
            "kernel", shape=[4, 2], initializer="glorot_uniform"
        )

    def call(self, inputs):
        kernel_init(self.kernel)
        at_init(inputs @ self.kernel)
        return inputs


def kernel_float64(k):
    assert k.shape == (4, 2)
    assert k.dtype == tf.float64


def matmul_float64(y):
    assert y.shape == (5, 2)
    assert y.dtype == tf.float64


def at_float64(y):
    assert y.shape == (5, 2)
    assert y.dtype == tf.float64


class Float64Model(tf.keras.Model):
    """Gives its weight an explicit `dtype`."""

    def __init__(self):
        super().__init__()
        self.kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel",
            shape=[input_shape[-1], 2],
            dtype=tf.float64,
            initializer="glorot_uniform",
        )

    def call(self, inputs):
        # The call casts a float input to the model's float32 compute dtype, so the other
        # operand is a float64 tensor of its own.
        x = tf.ones([5, 4], dtype=tf.float64)
        kernel_float64(self.kernel)
        matmul_float64(tf.matmul(x, self.kernel))
        at_float64(x @ self.kernel)
        return inputs


def kernel_initializer_call(k):
    assert k.shape == (4, 2)
    assert k.dtype == tf.float32


def at_initializer_call(y):
    assert y.shape == (5, 2)
    assert y.dtype == tf.float32


class InitializerCallModel(tf.keras.Model):
    """Passes an initializer that a call returns."""

    def __init__(self):
        super().__init__()
        self.kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel",
            shape=[input_shape[-1], 2],
            initializer=tf.random_normal_initializer(stddev=0.02),
        )

    def call(self, inputs):
        kernel_initializer_call(self.kernel)
        at_initializer_call(inputs @ self.kernel)
        return inputs


def kernel_no_initializer(k):
    assert k.shape == (4, 2)
    assert k.dtype == tf.float32


def at_no_initializer(y):
    assert y.shape == (5, 2)
    assert y.dtype == tf.float32


class NoInitializerModel(tf.keras.Model):
    """Passes no initializer."""

    def __init__(self):
        super().__init__()
        self.kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel", shape=[input_shape[-1], 2])

    def call(self, inputs):
        kernel_no_initializer(self.kernel)
        at_no_initializer(inputs @ self.kernel)
        return inputs


def kernel_keyword_name(k):
    assert k.shape == (4, 2)
    assert k.dtype == tf.float32


def at_keyword_name(y):
    assert y.shape == (5, 2)
    assert y.dtype == tf.float32


class KeywordNameModel(tf.keras.Model):
    """Passes the weight's name by keyword."""

    def __init__(self):
        super().__init__()
        self.kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel", shape=[input_shape[-1], 2], initializer="glorot_uniform"
        )

    def call(self, inputs):
        kernel_keyword_name(self.kernel)
        at_keyword_name(inputs @ self.kernel)
        return inputs


def kernel_regularizer(k):
    assert k.shape == (4, 2)
    assert k.dtype == tf.float32


def at_regularizer(y):
    assert y.shape == (5, 2)
    assert y.dtype == tf.float32


class RegularizerModel(tf.keras.Model):
    """Passes a regularizer."""

    def __init__(self):
        super().__init__()
        self.kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel",
            shape=[input_shape[-1], 2],
            initializer="glorot_uniform",
            regularizer=tf.keras.regularizers.l2(0.01),
        )

    def call(self, inputs):
        kernel_regularizer(self.kernel)
        at_regularizer(inputs @ self.kernel)
        return inputs


def kernel_nested(k):
    assert k.shape == (4, 2)
    assert k.dtype == tf.float32


def at_nested(y):
    assert y.shape == (5, 2)
    assert y.dtype == tf.float32


class InnerModel(tf.keras.Model):
    """Is constructed inside another layer's `__init__`."""

    def __init__(self):
        super().__init__()
        self.kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel", shape=[input_shape[-1], 2], initializer="glorot_uniform"
        )

    def call(self, inputs):
        kernel_nested(self.kernel)
        y = inputs @ self.kernel
        at_nested(y)
        return y


class OuterLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.inner = InnerModel()

    def call(self, inputs):
        return self.inner(inputs)


def kernel_layer(k):
    assert k.shape == (4, 2)
    assert k.dtype == tf.float32


def matmul_layer(y):
    assert y.shape == (5, 2)
    assert y.dtype == tf.float32


class BuildLayer(tf.keras.layers.Layer):
    """The control: a layer, not a model, creating its weight the same way."""

    def __init__(self):
        super().__init__()
        self.kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel", shape=[input_shape[-1], 2], initializer="glorot_uniform"
        )

    def call(self, inputs):
        kernel_layer(self.kernel)
        matmul_layer(tf.matmul(inputs, self.kernel))
        return inputs


BuildModel()(tf.random.uniform([5, 4]))
InitModel()(tf.random.uniform([5, 4]))
Float64Model()(tf.random.uniform([5, 4]))
InitializerCallModel()(tf.random.uniform([5, 4]))
NoInitializerModel()(tf.random.uniform([5, 4]))
KeywordNameModel()(tf.random.uniform([5, 4]))
RegularizerModel()(tf.random.uniform([5, 4]))
OuterLayer()(tf.random.uniform([5, 4]))
BuildLayer()(tf.random.uniform([5, 4]))
