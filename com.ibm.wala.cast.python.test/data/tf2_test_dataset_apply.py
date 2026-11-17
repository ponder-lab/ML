# The dataset `Dataset.apply` hands the function it applies, and the dataset it returns
# (wala/ML#1290).

import os
import tempfile

import tensorflow as tf


def hand_back(dataset):
    assert dataset.element_spec.shape == (3,)
    return dataset


def hand_back_other(dataset):
    return dataset


def consume_handed_back(dataset):
    assert dataset.element_spec.shape == (5,)


def batch_rows(dataset):
    assert dataset.element_spec.shape == (7,)
    return dataset.batch(2, drop_remainder=True)


def consume_batched(dataset):
    assert dataset.element_spec.shape == (2, 7)


def hand_back_lines(dataset):
    assert dataset.element_spec.shape == ()
    assert dataset.element_spec.dtype == tf.string
    return dataset


def consume_lines(dataset):
    assert dataset.element_spec.dtype == tf.string


def in_lambda(dataset):
    assert dataset.element_spec.shape == (11,)
    return dataset


def shapes_of(dataset):
    assert dataset.element_spec.shape == (13,)
    return tf.nest.map_structure(lambda spec: spec.shape, dataset.element_spec)


def make_batcher(size):
    return lambda dataset: dataset.padded_batch(
        size, padded_shapes=shapes_of(dataset), drop_remainder=True
    )


def consume_padded(x):
    assert x.shape == (2, 13) and x.dtype == tf.float32


def batch_window(dataset):
    assert dataset.element_spec.shape == (19,)
    return dataset.batch(2, drop_remainder=True)


def reduce_applied(key, window):
    return window.apply(batch_window)


def chained_first(dataset):
    assert dataset.element_spec.shape == (29,)
    return dataset.batch(2, drop_remainder=True)


def chained_second(dataset):
    assert dataset.element_spec.shape == (2, 29)
    return dataset


def called_and_applied(dataset):
    assert dataset.element_spec.shape in ((31,), (37,))
    return dataset


def features(line):
    tokens = tf.strings.split(line)
    return {"tokens": tokens, "length": tf.size(tokens)}


def hand_back_features(dataset):
    assert set(dataset.element_spec) == {"tokens", "length"}
    return dataset


def pass_features(dataset):
    assert set(dataset.element_spec) == {"tokens", "length"}


def consume_row(row):
    assert row.shape == (43,) and row.dtype == tf.float32


def iterate(dataset):
    for row in dataset:
        consume_row(row)
    return dataset


def consume_zipped(a, b):
    assert a.shape == (53,) and b.shape == (53,)


def zip_parameter(dataset):
    for a, b in tf.data.Dataset.zip((dataset, dataset)):
        consume_zipped(a, b)
    return dataset


def consume_zipped_passed(a, b):
    assert a.shape == (59,) and b.shape == (59,)


def zip_passed(dataset):
    for a, b in tf.data.Dataset.zip((dataset, dataset)):
        consume_zipped_passed(a, b)


def batch_directly(dataset):
    assert dataset.element_spec.shape == (47,)
    return dataset.batch(2, drop_remainder=True)


path = os.path.join(tempfile.mkdtemp(), "lines.txt")
with open(path, "w") as f:
    f.write("a b\nc d e\nf\n")

tf.data.Dataset.from_tensor_slices(tf.ones([4, 3])).apply(hand_back)

consume_handed_back(
    tf.data.Dataset.from_tensor_slices(tf.ones([4, 5])).apply(hand_back_other)
)

consume_batched(tf.data.Dataset.from_tensor_slices(tf.ones([4, 7])).apply(batch_rows))

consume_lines(tf.data.TextLineDataset(path).apply(hand_back_lines))

tf.data.Dataset.from_tensor_slices(tf.ones([4, 11])).apply(
    lambda dataset: in_lambda(dataset)
)

for x in tf.data.Dataset.from_tensor_slices(tf.ones([4, 13])).apply(make_batcher(2)):
    consume_padded(x)

for _ in tf.data.Dataset.from_tensor_slices(tf.ones([4, 19])).apply(
    tf.data.experimental.group_by_window(
        key_func=lambda row: tf.constant(0, tf.int64),
        reduce_func=reduce_applied,
        window_size=2,
    )
):
    pass

tf.data.Dataset.from_tensor_slices(tf.ones([4, 29])).apply(chained_first).apply(
    chained_second
)

called_and_applied(tf.data.Dataset.from_tensor_slices(tf.ones([4, 31])))
tf.data.Dataset.from_tensor_slices(tf.ones([4, 37])).apply(called_and_applied)

tf.data.TextLineDataset(path).map(features).apply(hand_back_features)
pass_features(tf.data.TextLineDataset(path).map(features))

tf.data.Dataset.from_tensor_slices(tf.ones([4, 43])).apply(iterate)

batch_directly(tf.data.Dataset.from_tensor_slices(tf.ones([4, 47])))

tf.data.Dataset.from_tensor_slices(tf.ones([4, 53])).apply(zip_parameter)
zip_passed(tf.data.Dataset.from_tensor_slices(tf.ones([4, 59])))
