# Test for wala/ML#993 (the encoder chain): values reaching a Keras layer's `call` along a training
# step arrive untyped. The step is a `tf.function` whose `input_signature` is the dataset's
# `element_spec` (dicts of specs); an inputter layer embeds `features["ids"]`; an encoder layer with
# its own `__call__`, which delegates to `super().__call__`, receives the embedded inputs beside a
# length and `training=True`, and its `call` builds a mask from them. A sink at each hop locates
# where the type is lost; a plain encoder without its own `__call__` is the control for the hop
# through `super().__call__`.
import tensorflow as tf


def consume_features_ids(x):
    pass


def consume_embedded(x):
    pass


def consume_step_inputs(x):
    pass


def consume_dunder_call_inputs(x):
    pass


def consume_call_inputs(x):
    pass


def consume_mask_inputs(x):
    pass


def consume_plain_call_inputs(x):
    pass


def consume_direct_dunder_call_inputs(x):
    pass


def consume_inherited_inputs(x):
    pass


def consume_settable_inputs(x):
    pass


def consume_settable_fn_out(x):
    pass


def consume_setter_value(x):
    pass


def passthrough(x):
    return x


def consume_property_inputs(x):
    pass


def consume_getattr_inputs(x):
    pass


class WordEmbedder(tf.keras.layers.Layer):
    def __init__(self, vocabulary_size, embedding_size):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size

    def build(self, input_shape):
        self.embedding = self.add_weight(
            "embedding", shape=[self.vocabulary_size, self.embedding_size]
        )
        super().build(input_shape)

    def call(self, features, training=None):
        ids = features["ids"]
        consume_features_ids(ids)
        outputs = tf.nn.embedding_lookup(self.embedding, ids)
        consume_embedded(outputs)
        return outputs


class Encoder(tf.keras.layers.Layer):
    def build_mask(self, inputs, sequence_length=None, dtype=tf.bool):
        consume_mask_inputs(inputs)
        if sequence_length is None:
            return None
        return tf.sequence_mask(
            sequence_length, maxlen=tf.shape(inputs)[1], dtype=dtype
        )

    def __call__(self, inputs, sequence_length=None, **kwargs):
        consume_dunder_call_inputs(inputs)
        outputs, state, sequence_length = super().__call__(
            inputs, sequence_length=sequence_length, **kwargs
        )
        return outputs, state, sequence_length


class SelfAttentionEncoder(Encoder):
    def __init__(self, num_units):
        super().__init__()
        self.num_units = num_units

    def call(self, inputs, sequence_length=None, training=None):
        consume_call_inputs(inputs)
        inputs *= self.num_units**0.5
        mask = self.build_mask(inputs, sequence_length=sequence_length)
        return inputs, None, sequence_length


class DirectCallEncoder(tf.keras.layers.Layer):
    # Control for the inherited `__call__`: the same `__call__` defined on the class itself.
    def __call__(self, inputs, sequence_length=None, **kwargs):
        consume_direct_dunder_call_inputs(inputs)
        return super().__call__(inputs, sequence_length=sequence_length, **kwargs)

    def call(self, inputs, sequence_length=None, training=None):
        return inputs, None, sequence_length


class PlainEncoder(tf.keras.layers.Layer):
    def call(self, inputs, sequence_length=None, training=None):
        consume_plain_call_inputs(inputs)
        return inputs, None, sequence_length


class ExampleInputter:
    def __init__(self, features_inputter):
        self.features_inputter = features_inputter


class BaseModel:
    # A property declared on a base class and read on the subclass's instance.
    @property
    def inherited_inputter(self):
        return self.examples_inputter.features_inputter


class Model(BaseModel):
    def __init__(self):
        self.features_inputter = WordEmbedder(100, 8)
        self.examples_inputter = ExampleInputter(self.features_inputter)
        self.encoder = SelfAttentionEncoder(8)
        self.plain_encoder = PlainEncoder()
        self.direct_encoder = DirectCallEncoder()
        # Assigned through the property's setter.
        self.settable_inputter = self.features_inputter
        self.settable_fn = passthrough

    # A property with a setter of the same name: the getter's value, not the setter, is what the
    # instance's attribute holds.
    @property
    def settable_inputter(self):
        return self._settable

    @settable_inputter.setter
    def settable_inputter(self, value):
        self._settable = value

    # A property whose setter sinks what it is given. The analysis does not run a setter: the
    # assignment in `__init__` reaches the attribute directly. The sink is called once below with a
    # function, so it is present; a call through the attribute must not reach it with the tensor.
    @property
    def settable_fn(self):
        return self._fn

    @settable_fn.setter
    def settable_fn(self, value):
        consume_setter_value(value)
        self._fn = value

    # The inputter reached through a property, as a sequence-to-sequence model exposes it.
    @property
    def property_inputter(self):
        return self.examples_inputter.features_inputter

    # The same through `getattr` with a default, the property's actual body in that model.
    @property
    def getattr_inputter(self):
        return getattr(
            self.examples_inputter, "features_inputter", self.examples_inputter
        )


model = Model()
consume_setter_value(passthrough)

dataset = tf.data.Dataset.from_tensor_slices(
    (
        {"ids": tf.constant([[1, 2, 3], [4, 5, 6]]), "length": tf.constant([3, 3])},
        {"ids": tf.constant([[7, 8, 9], [1, 2, 3]]), "length": tf.constant([3, 3])},
    )
).batch(2)


@tf.function(input_signature=dataset.element_spec)
def training_step(source, target):
    source_inputs = model.features_inputter(source, training=True)
    consume_step_inputs(source_inputs)
    encoder_outputs, _, _ = model.encoder(
        source_inputs, source["length"], training=True
    )
    plain_outputs, _, _ = model.plain_encoder(
        source_inputs, source["length"], training=True
    )
    direct_outputs, _, _ = model.direct_encoder(
        source_inputs, source["length"], training=True
    )
    property_inputs = model.property_inputter(source, training=True)
    consume_property_inputs(property_inputs)
    getattr_inputs = model.getattr_inputter(source, training=True)
    consume_getattr_inputs(getattr_inputs)
    inherited_inputs = model.inherited_inputter(source, training=True)
    consume_inherited_inputs(inherited_inputs)
    settable_inputs = model.settable_inputter(source, training=True)
    consume_settable_inputs(settable_inputs)
    fn_out = model.settable_fn(source_inputs)
    consume_settable_fn_out(fn_out)
    return encoder_outputs


for source, target in dataset:
    out = training_step(source, target)
    assert out.shape == (2, 3, 8) and out.dtype == tf.float32
    assert source["ids"].shape == (2, 3) and source["ids"].dtype == tf.int32
