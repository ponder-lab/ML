# `fit` on a dataset chosen by a branch and then transformed, `(a if flag else b).batch(2)`, with
# Keras's default `train_step`. The transformation's receiver is either dataset, so the inputs
# component the model's `call` receives is either dataset's component, batched: a tuple of tensors
# where the dataset's inputs component is a tuple, and a tensor where it is flat.
import tensorflow as tf


def loss(y, p):
    return tf.reduce_mean(p)


def fit(model, dataset):
    model.compile(optimizer="sgd", loss=loss)
    model.fit(dataset, epochs=1, verbose=0)


def pair_dataset():
    return tf.data.Dataset.from_tensor_slices(
        ((tf.ones([8, 4]), tf.ones([8, 3])), tf.zeros([8]))
    )


def other_pair_dataset():
    return tf.data.Dataset.from_tensor_slices(
        ((tf.ones([8, 4]), tf.ones([8, 5])), tf.zeros([8]))
    )


def flat_dataset():
    return tf.data.Dataset.from_tensor_slices((tf.ones([8, 6]), tf.zeros([8])))


def other_flat_dataset():
    return tf.data.Dataset.from_tensor_slices((tf.ones([8, 7]), tf.zeros([8])))


# A tuple component on one dataset and a flat one on the other.
def consume_mixed_batched(inputs):
    assert isinstance(inputs, (tuple, tf.Tensor))
    if isinstance(inputs, tf.Tensor):
        assert inputs.shape[1:] == (6,) and inputs.dtype == tf.float32


class MixedBatched(tf.keras.Model):
    def call(self, inputs):
        consume_mixed_batched(inputs)
        return inputs[0]


def fit_mixed_batched(flag):
    fit(MixedBatched(), (pair_dataset() if flag else flat_dataset()).batch(2))


# One call site, so one frame where the choice is either dataset.
for flag in (True, False):
    fit_mixed_batched(flag)


# A tuple component on both datasets.
def consume_pair_batched(inputs):
    assert isinstance(inputs, tuple) and len(inputs) == 2


class PairBatched(tf.keras.Model):
    def call(self, inputs):
        consume_pair_batched(inputs)
        x, m = inputs
        return x


def fit_pair_batched(flag):
    fit(PairBatched(), (pair_dataset() if flag else other_pair_dataset()).batch(2))


for flag in (True, False):
    fit_pair_batched(flag)


# A flat component on both datasets, of different widths.
def consume_flat_batched(inputs):
    assert isinstance(inputs, tf.Tensor)
    assert inputs.shape[1:] in ((6,), (7,)) and inputs.dtype == tf.float32


class FlatBatched(tf.keras.Model):
    def call(self, inputs):
        consume_flat_batched(inputs)
        return inputs


def fit_flat_batched(flag):
    fit(FlatBatched(), (flat_dataset() if flag else other_flat_dataset()).batch(2))


for flag in (True, False):
    fit_flat_batched(flag)


# A tuple component on one dataset and a flat one on the other, under two transformations.
def consume_mixed_shuffled(inputs):
    assert isinstance(inputs, (tuple, tf.Tensor))
    if isinstance(inputs, tf.Tensor):
        assert inputs.shape[1:] == (6,) and inputs.dtype == tf.float32


class MixedShuffled(tf.keras.Model):
    def call(self, inputs):
        consume_mixed_shuffled(inputs)
        return inputs[0]


def fit_mixed_shuffled(flag):
    fit(
        MixedShuffled(),
        (pair_dataset() if flag else flat_dataset()).shuffle(4).batch(2),
    )


for flag in (True, False):
    fit_mixed_shuffled(flag)


def triple_dataset():
    return tf.data.Dataset.from_tensor_slices(
        ((tf.ones([8, 4]), tf.ones([8, 3]), tf.ones([8, 2])), tf.zeros([8]))
    )


# A tuple component on both datasets, of different arities, under two transformations.
def consume_pair_or_triple_shuffled(inputs):
    assert isinstance(inputs, tuple) and len(inputs) in (2, 3)


class PairOrTripleShuffled(tf.keras.Model):
    def call(self, inputs):
        consume_pair_or_triple_shuffled(inputs)
        return inputs[0]


def fit_pair_or_triple_shuffled(flag):
    fit(
        PairOrTripleShuffled(),
        (pair_dataset() if flag else triple_dataset()).shuffle(4).batch(2),
    )


for flag in (True, False):
    fit_pair_or_triple_shuffled(flag)


# As the mixed case under two transformations, with the flat dataset first.
def consume_flat_or_pair_shuffled(inputs):
    assert isinstance(inputs, (tuple, tf.Tensor))
    if isinstance(inputs, tf.Tensor):
        assert inputs.shape[1:] == (6,) and inputs.dtype == tf.float32


class FlatOrPairShuffled(tf.keras.Model):
    def call(self, inputs):
        consume_flat_or_pair_shuffled(inputs)
        return inputs[0]


def fit_flat_or_pair_shuffled(flag):
    fit(
        FlatOrPairShuffled(),
        (flat_dataset() if flag else pair_dataset()).shuffle(4).batch(2),
    )


for flag in (True, False):
    fit_flat_or_pair_shuffled(flag)


# A `map` whose function takes the element's components as parameters and returns the first, a
# tuple, so the mapped element is that tuple and its inputs component is a tensor.
def consume_map_component(inputs):
    assert isinstance(inputs, tf.Tensor)
    assert inputs.shape[1:] == (4,) and inputs.dtype == tf.float32


class MapComponent(tf.keras.Model):
    def call(self, inputs):
        consume_map_component(inputs)
        return inputs


fit(MapComponent(), pair_dataset().map(lambda inp, y: inp).batch(2))
