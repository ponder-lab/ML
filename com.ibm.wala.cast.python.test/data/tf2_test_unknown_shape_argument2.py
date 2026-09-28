# Test https://github.com/wala/ML/issues/978: a generator argument whose shape is unknown must
# degrade the result, not crash the analysis. `tf.where` over a runtime comparison has a
# data-dependent row count, which the analysis reads as an unknown shape. These are the other
# generators that read such an argument, kept apart from the ragged ones so each fixture's
# first crash does not mask the rest.
import tensorflow as tf


def consume_flatten(x):
    pass


def consume_poisson(x):
    pass


def consume_dataset_element(x):
    pass


def consume_chosen_element(x):
    pass


def consume_sampled_element(x):
    pass


tokens = tf.constant([1, 2, 1, 3])
# Positions of the ones: data-dependent length, so its shape is unknown to the analysis.
starts = tf.squeeze(tf.where(tf.equal(tokens, 1)), -1)
assert starts.shape == (2,) and starts.dtype == tf.int64

flat = tf.keras.layers.Flatten()(tf.cast(starts, tf.float32))
assert flat.shape == (2, 1)
consume_flatten(flat)

p = tf.random.poisson([2], tf.cast(starts, tf.float32))
assert p.shape == (2, 2)
consume_poisson(p)

for element in tf.data.Dataset.from_tensors(starts):
    consume_dataset_element(element)

chosen = tf.data.Dataset.choose_from_datasets(
    [tf.data.Dataset.from_tensors(starts)], tf.data.Dataset.range(1)
)
for element in chosen:
    assert element.shape == (2,)
    consume_chosen_element(element)

sampled = tf.data.Dataset.sample_from_datasets(
    [tf.data.Dataset.from_tensors(starts)], weights=[1.0]
)
for element in sampled:
    assert element.shape == (2,)
    consume_sampled_element(element)
