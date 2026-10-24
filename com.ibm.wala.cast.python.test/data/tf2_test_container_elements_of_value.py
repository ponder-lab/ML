# The elements `ContainerElements.ofValue` reads for values that may be no container instance: the
# structured `inputs` a Keras step hands to `call` when `fit` is given a dataset, a flat one, one
# that is a tuple on one branch and a tensor on the other, and an ordinary local tuple. `fit` traces
# `call` with the batch axis unspecified, so the sinks check the other axes.
import tensorflow as tf


def loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)


def fit(model, dataset):
    model.compile(optimizer="sgd", loss=loss)
    model.fit(dataset, epochs=1, verbose=0)


def consume_pair(inputs):
    assert isinstance(inputs, tuple) and len(inputs) == 2
    assert inputs[0].shape[1:] == (4,) and inputs[0].dtype == tf.float32
    assert inputs[1].shape[1:] == (3,) and inputs[1].dtype == tf.float32


class Pair(tf.keras.Model):
    def call(self, inputs):
        consume_pair(inputs)
        return inputs[0]


fit(
    Pair(),
    tf.data.Dataset.from_tensor_slices(
        ((tf.ones([8, 4]), tf.ones([8, 3])), tf.zeros([8]))
    ).batch(2),
)


def consume_keyed(inputs):
    assert isinstance(inputs, dict) and sorted(inputs) == ["ids", "m"]
    assert inputs["ids"].shape[1:] == (3,) and inputs["ids"].dtype == tf.int32
    assert inputs["m"].shape[1:] == (5,) and inputs["m"].dtype == tf.float32


class Keyed(tf.keras.Model):
    def call(self, inputs):
        consume_keyed(inputs)
        return inputs["m"]


fit(
    Keyed(),
    tf.data.Dataset.from_tensor_slices(
        ({"ids": tf.ones([8, 3], tf.int32), "m": tf.ones([8, 5])}, tf.zeros([8]))
    ).batch(2),
)


def triple_batches():
    yield (tf.ones([2, 4]), tf.ones([2, 3]), tf.ones([2, 5], tf.int32)), tf.zeros([2])


def consume_triple(inputs):
    assert isinstance(inputs, tuple) and len(inputs) == 3
    assert inputs[0].shape[1:] == (4,) and inputs[0].dtype == tf.float32
    assert inputs[1].shape[1:] == (3,) and inputs[1].dtype == tf.float32
    assert inputs[2].shape[1:] == (5,) and inputs[2].dtype == tf.int32


class Triple(tf.keras.Model):
    def call(self, inputs):
        consume_triple(inputs)
        return inputs[0]


fit(
    Triple(),
    tf.data.Dataset.from_generator(
        triple_batches,
        output_signature=(
            (
                tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 5), dtype=tf.int32),
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        ),
    ),
)


def pair_batches():
    yield (tf.ones([2, 4]), tf.ones([2, 3])), tf.zeros([2])


def consume_pair_from_generator(inputs):
    assert isinstance(inputs, tuple) and len(inputs) == 2
    assert inputs[0].shape[1:] == (4,) and inputs[0].dtype == tf.float32
    assert inputs[1].shape[1:] == (3,) and inputs[1].dtype == tf.float32


class PairFromGenerator(tf.keras.Model):
    def call(self, inputs):
        consume_pair_from_generator(inputs)
        return inputs[0]


fit(
    PairFromGenerator(),
    tf.data.Dataset.from_generator(
        pair_batches,
        output_types=((tf.float32, tf.float32), tf.float32),
        output_shapes=(
            (tf.TensorShape([None, 4]), tf.TensorShape([None, 3])),
            tf.TensorShape([None]),
        ),
    ),
)


def pairs():
    yield tf.ones([2, 4]), tf.ones([2, 3])


def consume_zipped_pair(inputs):
    # The first zipped dataset's whole element, a pair `from_generator` declares.
    assert isinstance(inputs, tuple) and len(inputs) == 2
    assert inputs[1].shape[1:] == (3,) and inputs[1].dtype == tf.float32


class ZippedPair(tf.keras.Model):
    def call(self, inputs):
        consume_zipped_pair(inputs)
        return inputs[0]


fit(
    ZippedPair(),
    tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_generator(
                pairs,
                output_signature=(
                    tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                    tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
                ),
            ),
            tf.data.Dataset.from_tensor_slices(tf.zeros([8])).batch(2),
        )
    ),
)


def consume_zipped_tensors(inputs):
    # The first zipped dataset's whole element, a pair `from_tensors` is given.
    assert isinstance(inputs, tuple) and len(inputs) == 2
    assert inputs[0].shape == (2, 4) and inputs[0].dtype == tf.float32
    assert inputs[1].shape == (2, 3) and inputs[1].dtype == tf.float32


class ZippedTensors(tf.keras.Model):
    def call(self, inputs):
        consume_zipped_tensors(inputs)
        return inputs[0]


fit(
    ZippedTensors(),
    tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_tensors((tf.ones([2, 4]), tf.ones([2, 3]))),
            tf.data.Dataset.from_tensors(tf.zeros([2])),
        )
    ),
)


def consume_nested(inputs):
    # Its first member is itself a tuple.
    assert isinstance(inputs, tuple) and len(inputs) == 2
    assert isinstance(inputs[0], tuple) and len(inputs[0]) == 2
    assert inputs[1].shape[1:] == (5,) and inputs[1].dtype == tf.float32


class Nested(tf.keras.Model):
    def call(self, inputs):
        consume_nested(inputs)
        return inputs[1]


fit(
    Nested(),
    tf.data.Dataset.from_tensor_slices(
        (((tf.ones([8, 4]), tf.ones([8, 3])), tf.ones([8, 5])), tf.zeros([8]))
    ).batch(2),
)


def consume_starred(inputs):
    assert isinstance(inputs, tuple) and len(inputs) == 3


class Starred(tf.keras.Model):
    def call(self, inputs):
        consume_starred(inputs)
        return inputs[2]


xs = [tf.ones([8, 4]), tf.ones([8, 3])]
fit(
    Starred(),
    tf.data.Dataset.from_tensor_slices(((*xs, tf.ones([8, 5])), tf.zeros([8]))).batch(
        2
    ),
)


def consume_flat_mapped(inputs):
    assert isinstance(inputs, tuple) and len(inputs) == 2


class FlatMapped(tf.keras.Model):
    def call(self, inputs):
        consume_flat_mapped(inputs)
        return inputs[0]


fit(
    FlatMapped(),
    tf.data.Dataset.from_tensor_slices(
        (tf.ones([8, 4]), tf.ones([8, 3]), tf.zeros([8]))
    )
    .flat_map(lambda a, b, y: tf.data.Dataset.from_tensors(((a, b), y)))
    .batch(2),
)


def consume_built(inputs):
    # A tuple of two at run time, built by `tuple` rather than written as a display.
    assert isinstance(inputs, tuple) and len(inputs) == 2


class Built(tf.keras.Model):
    def call(self, inputs):
        consume_built(inputs)
        return inputs[0]


parts = [tf.ones([8, 4]), tf.ones([8, 3])]
fit(Built(), tf.data.Dataset.from_tensor_slices((tuple(parts), tf.zeros([8]))).batch(2))


def consume_mixed(inputs):
    # A tuple when the structured dataset is chosen and a tensor when the flat one is.
    assert isinstance(inputs, (tuple, tf.Tensor))


class Mixed(tf.keras.Model):
    def call(self, inputs):
        consume_mixed(inputs)
        return inputs[0]


def pair_dataset():
    return tf.data.Dataset.from_tensor_slices(
        ((tf.ones([8, 4]), tf.ones([8, 3])), tf.zeros([8]))
    )


def flat_dataset():
    return tf.data.Dataset.from_tensor_slices((tf.ones([8, 6]), tf.zeros([8])))


def fit_mixed(flag):
    fit(Mixed(), pair_dataset().batch(2) if flag else flat_dataset().batch(2))


# One call site, so one frame where the choice is either dataset.
for flag in (True, False):
    fit_mixed(flag)


def consume_mixed_batched(inputs):
    # As `consume_mixed`, with the batching applied to the chosen dataset.
    assert isinstance(inputs, (tuple, tf.Tensor))


class MixedBatched(tf.keras.Model):
    def call(self, inputs):
        consume_mixed_batched(inputs)
        return inputs[0]


def fit_mixed_batched(flag):
    fit(MixedBatched(), (pair_dataset() if flag else flat_dataset()).batch(2))


for flag in (True, False):
    fit_mixed_batched(flag)


def triple_dataset():
    return tf.data.Dataset.from_tensor_slices(
        ((tf.ones([8, 4]), tf.ones([8, 3]), tf.ones([8, 5])), tf.zeros([8]))
    )


def other_pair_dataset():
    return tf.data.Dataset.from_tensor_slices(
        ((tf.ones([8, 4]), tf.ones([8, 5])), tf.zeros([8]))
    )


def consume_lengths_shuffled(inputs):
    # A pair when the pair dataset is chosen and a triple when the triple one is.
    assert isinstance(inputs, tuple) and len(inputs) in (2, 3)


class LengthsShuffled(tf.keras.Model):
    def call(self, inputs):
        consume_lengths_shuffled(inputs)
        return inputs[0]


def fit_lengths_shuffled(flag):
    fit(
        LengthsShuffled(),
        (pair_dataset() if flag else triple_dataset()).shuffle(4).batch(2),
    )


for flag in (True, False):
    fit_lengths_shuffled(flag)


def consume_mixed_shuffled(inputs):
    # A tensor when the flat dataset is chosen and a tuple when the pair one is.
    assert isinstance(inputs, (tuple, tf.Tensor))


class MixedShuffled(tf.keras.Model):
    def call(self, inputs):
        consume_mixed_shuffled(inputs)
        return inputs[0]


def fit_mixed_shuffled(flag):
    fit(
        MixedShuffled(),
        (flat_dataset() if flag else pair_dataset()).shuffle(4).batch(2),
    )


for flag in (True, False):
    fit_mixed_shuffled(flag)


def consume_agreeing(inputs):
    # A pair either way, whose second member is (None, 3) or (None, 5).
    assert isinstance(inputs, tuple) and len(inputs) == 2
    assert inputs[0].shape[1:] == (4,) and inputs[0].dtype == tf.float32
    assert inputs[1].shape[1:] in ((3,), (5,)) and inputs[1].dtype == tf.float32


class Agreeing(tf.keras.Model):
    def call(self, inputs):
        consume_agreeing(inputs)
        return inputs[0]


def fit_agreeing(flag):
    fit(Agreeing(), (pair_dataset() if flag else other_pair_dataset()).batch(2))


for flag in (True, False):
    fit_agreeing(flag)


def consume_mapped_pair(inputs):
    # `map` keeps the element's `((x, mask), w)`, so `fit` takes `(x, mask)` for the inputs.
    assert isinstance(inputs, tuple) and len(inputs) == 2
    assert inputs[0].shape[1:] == (4,) and inputs[0].dtype == tf.float32
    assert inputs[1].shape[1:] == (3,) and inputs[1].dtype == tf.float32


class MappedPair(tf.keras.Model):
    def call(self, inputs):
        consume_mapped_pair(inputs)
        return inputs[0]


fit(
    MappedPair(),
    tf.data.Dataset.from_tensor_slices(
        (((tf.ones([8, 4]), tf.ones([8, 3])), tf.ones([8, 2])), tf.zeros([8]))
    )
    .map(lambda inp, y: inp)
    .batch(2),
)


def consume_loop_lengths(inputs):
    # A pair when the pair dataset is chosen and a triple when the triple one is.
    assert isinstance(inputs, tuple) and len(inputs) in (2, 3)


def loop_lengths(flag):
    for inputs, y in (pair_dataset() if flag else triple_dataset()).batch(2):
        consume_loop_lengths(inputs)


for flag in (True, False):
    loop_lengths(flag)


def consume_loop_agreeing(inputs):
    assert isinstance(inputs, tuple) and len(inputs) == 2


def loop_agreeing(flag):
    for inputs, y in (pair_dataset() if flag else other_pair_dataset()).batch(2):
        consume_loop_agreeing(inputs)


for flag in (True, False):
    loop_agreeing(flag)


def consume_mapped_inputs(inputs):
    # `map` keeps only the element's pair, so `fit` takes its first member for the inputs.
    assert isinstance(inputs, tf.Tensor)
    assert inputs.shape[1:] == (4,) and inputs.dtype == tf.float32


class MappedInputs(tf.keras.Model):
    def call(self, inputs):
        consume_mapped_inputs(inputs)
        return inputs


fit(MappedInputs(), pair_dataset().map(lambda inp, y: inp).batch(2))


def consume_flat(inputs):
    assert isinstance(inputs, tf.Tensor)
    assert inputs.shape[1:] == (4,) and inputs.dtype == tf.float32


class Flat(tf.keras.Model):
    def call(self, inputs):
        consume_flat(inputs)
        return inputs


fit(
    Flat(),
    tf.data.Dataset.from_tensor_slices((tf.ones([8, 4]), tf.zeros([8]))).batch(2),
)


def consume_local(t):
    assert isinstance(t, tuple) and len(t) == 2
    assert t[0].shape == (2, 3) and t[0].dtype == tf.float32


consume_local((tf.ones([2, 3]), 1))


def consume_local_join(t):
    # Either of two pairs, so a length of 2 with each position either pair's.
    assert isinstance(t, tuple) and len(t) == 2
    assert t[0].shape in ((2, 3), (2, 5)) and t[0].dtype == tf.float32
    assert t[1].shape in ((2, 4), (2, 6)) and t[1].dtype == tf.float32


def local_join(flag):
    consume_local_join(
        (tf.ones([2, 3]), tf.ones([2, 4]))
        if flag
        else (tf.zeros([2, 5]), tf.ones([2, 6]))
    )


for flag in (True, False):
    local_join(flag)


def consume_local_lengths(t):
    # A pair or a triple, so no one length.
    assert isinstance(t, tuple) and len(t) in (2, 3)
    assert t[0].shape == (2, 3) and t[0].dtype == tf.float32


def local_lengths(flag):
    consume_local_lengths(
        (tf.ones([2, 3]), tf.ones([2, 4]))
        if flag
        else (tf.ones([2, 3]), tf.ones([2, 4]), tf.ones([2, 5]))
    )


for flag in (True, False):
    local_lengths(flag)


def consume_mapped_built(inputs):
    # `map` builds `((x, x * 2.0), y)`, so `fit` takes the built pair for the inputs.
    assert isinstance(inputs, tuple) and len(inputs) == 2
    assert inputs[0].shape[1:] == (4,) and inputs[0].dtype == tf.float32
    assert inputs[1].shape[1:] == (4,) and inputs[1].dtype == tf.float32


class MappedBuilt(tf.keras.Model):
    def call(self, inputs):
        consume_mapped_built(inputs)
        return inputs[0]


fit(
    MappedBuilt(),
    tf.data.Dataset.from_tensor_slices((tf.ones([8, 4]), tf.zeros([8])))
    .map(lambda x, y: ((x, x * 2.0), y))
    .batch(2),
)
