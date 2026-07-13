# Fixtures for wala/ML#725: the rejection and alternate arms of the `build` stored-attribute
# resolution helpers. Every targeted stored value derives from an `input_shape` subscript, whose
# result has an empty points-to set, so resolution must go through the helper chain rather than
# the constant-propagation path. Each layer routes its weight through a per-case sink so the
# JUnit test can pin the resolved (or deliberately degraded) weight type.
import tensorflow as tf


class Box:
    pass


def apply_attr_read(x, w):
    return tf.matmul(x, w)


def apply_dict_key(x, w):
    return tf.matmul(x, w)


def apply_opaque_index(x, w):
    return tf.matmul(x, w)


def apply_oob_index(x, w):
    return tf.matmul(x, w)


def apply_negative_oob_index(x, w):
    return tf.matmul(x, w)


def apply_nested_cfg(x, w):
    return tf.matmul(x, w)


def apply_direct_nested_cfg(x, w):
    return tf.matmul(x, w)


class AttrReadLayer(tf.keras.layers.Layer):
    # The stored value is an attribute read off a same-body holder object: the subscript
    # resolver must classify the non-numeric string member ("size") as an attribute read, not an
    # index, and the same-body write chase then recovers the subscript-derived value.
    def build(self, input_shape):
        box = Box()
        box.size = input_shape[1]
        self.hidden_size = box.size
        self.w = self.add_weight(
            name="kernel", shape=[self.hidden_size, 5], trainable=True
        )
        self.built = True

    def call(self, input_tensor):
        return apply_attr_read(input_tensor, self.w)


class DictKeyLayer(tf.keras.layers.Layer):
    # The stored value is a dict subscript whose key is the numeric string "2": subscript
    # indices can surface as string constants, so the parse must succeed, the dict must be
    # rejected as a shape vector, and the same-body write chase then recovers the
    # subscript-derived value.
    def build(self, input_shape):
        table = {"2": input_shape[1]}
        self.hidden_size = table["2"]
        self.w = self.add_weight(
            name="kernel", shape=[self.hidden_size, 5], trainable=True
        )
        self.built = True

    def call(self, input_tensor):
        return apply_dict_key(input_tensor, self.w)


class OpaqueIndexLayer(tf.keras.layers.Layer):
    # The subscript index is a runtime int the analysis cannot resolve, so the subscript
    # resolver's constant-index sentinel guard must reject it.
    def __init__(self, idx, **kwargs):
        super(OpaqueIndexLayer, self).__init__(**kwargs)
        self.idx = idx

    def build(self, input_shape):
        self.hidden_size = input_shape[self.idx]
        self.w = self.add_weight(
            name="kernel", shape=[self.hidden_size, 5], trainable=True
        )
        self.built = True

    def call(self, input_tensor):
        return apply_opaque_index(input_tensor, self.w)


class OobIndexLayer(tf.keras.layers.Layer):
    # The first write's subscript index is beyond the resolved rank (a runtime-guarded
    # `try`/`except` read; a TensorShape subscript raises IndexError through its dimension
    # list), so the bounds check rejects it and the unresolvable write makes the attribute
    # unresolvable.
    def build(self, input_shape):
        try:
            self.hidden_size = input_shape[5]
        except IndexError:
            self.hidden_size = input_shape[1]
        self.w = self.add_weight(
            name="kernel", shape=[self.hidden_size, 5], trainable=True
        )
        self.built = True

    def call(self, input_tensor):
        return apply_oob_index(input_tensor, self.w)


class NegativeOobIndexLayer(tf.keras.layers.Layer):
    # Negative-index companion of OobIndexLayer: the normalized index falls below zero.
    def build(self, input_shape):
        try:
            self.hidden_size = input_shape[-5]
        except IndexError:
            self.hidden_size = input_shape[-1]
        self.w = self.add_weight(
            name="kernel", shape=[self.hidden_size, 5], trainable=True
        )
        self.built = True

    def call(self, input_tensor):
        return apply_negative_oob_index(input_tensor, self.w)


class NestedCfgLayer(tf.keras.layers.Layer):
    # The stored value is a member of a class declared in the method body (a config-class
    # idiom); the member's class-declaration write is a field-put on the class object's local,
    # and its value derives from an `input_shape` subscript through the closure.
    def build(self, input_shape):
        class Cfg:
            size = input_shape[1]

        self.hidden_size = Cfg.size
        self.w = self.add_weight(
            name="kernel", shape=[self.hidden_size, 5], trainable=True
        )
        self.built = True

    def call(self, input_tensor):
        return apply_nested_cfg(input_tensor, self.w)


class DirectNestedCfgLayer(tf.keras.layers.Layer):
    # Companion of NestedCfgLayer that reads the config member directly in the weight shape,
    # with no intermediate stored attribute.
    def build(self, input_shape):
        class Cfg:
            size = input_shape[1]

        self.w = self.add_weight(name="kernel", shape=[Cfg.size, 5], trainable=True)
        self.built = True

    def call(self, input_tensor):
        return apply_direct_nested_cfg(input_tensor, self.w)


x = tf.ones((2, 6))

attr_read_layer = AttrReadLayer()
out1 = attr_read_layer(x)
assert out1.shape == (2, 5)
assert out1.dtype == tf.float32

dict_key_layer = DictKeyLayer()
out2 = dict_key_layer(x)
assert out2.shape == (2, 5)
assert out2.dtype == tf.float32

opaque_index_layer = OpaqueIndexLayer(int("1"))
out3 = opaque_index_layer(x)
assert out3.shape == (2, 5)
assert out3.dtype == tf.float32

oob_index_layer = OobIndexLayer()
out4 = oob_index_layer(x)
assert out4.shape == (2, 5)
assert out4.dtype == tf.float32

negative_oob_index_layer = NegativeOobIndexLayer()
out5 = negative_oob_index_layer(x)
assert out5.shape == (2, 5)
assert out5.dtype == tf.float32

nested_cfg_layer = NestedCfgLayer()
out8 = nested_cfg_layer(x)
assert out8.shape == (2, 5)
assert out8.dtype == tf.float32

direct_nested_cfg_layer = DirectNestedCfgLayer()
out9 = direct_nested_cfg_layer(x)
assert out9.shape == (2, 5)
assert out9.dtype == tf.float32
