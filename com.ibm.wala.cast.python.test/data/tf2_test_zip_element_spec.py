# The `element_spec` of a `tf.data.Dataset.zip` whose arguments have elements that are not
# mirrored: a `map` over a file's lines (wala/ML#1280). The spec has the nesting of the datasets the
# zip was given, with each dataset's own spec in its place.

import os
import tempfile

import tensorflow as tf


def consume_spec(spec):
    pass


def consume_member_spec(spec):
    pass


def consume_leaf_spec(spec):
    pass


def consume_leaf_shape(shape):
    pass


def consume_shapes(shapes):
    pass


def consume_shapes_member(shapes):
    pass


def consume_shape_leaf(shape):
    pass


def consume_tensor_spec(spec):
    pass


def consume_tensor_shape(shape):
    pass


def consume_mixed_mapped_spec(spec):
    pass


def consume_mixed_mirrored_spec(spec):
    pass


def consume_mixed_mirrored_leaf_spec(spec):
    pass


def consume_mixed_root_spec(spec):
    pass


def consume_mixed_shape_leaf(shape):
    pass


def consume_roots_spec(spec):
    pass


def consume_roots_member_spec(spec):
    pass


def consume_mirrored_spec(spec):
    pass


def consume_mirrored_leaf_spec(spec):
    pass


def consume_nested_leaf_spec(spec):
    pass


def consume_keyed_spec(spec):
    pass


def consume_keyed_leaf_spec(spec):
    pass


def consume_shuffled_leaf_spec(spec):
    pass


def consume_batched_leaf_shape(shape):
    pass


def consume_applied_shapes_member(shapes):
    pass


def consume_applied_shape_leaf(shape):
    pass


def use_leaf_shape(t):
    assert t.shape == (5,)
    assert t.dtype == tf.float32


def use_other_leaf_shape(t):
    assert t.shape == (7,)
    assert t.dtype == tf.float32


def use_leaf_dtype(t):
    assert t.shape == (2,)
    assert t.dtype == tf.int32


def use_row(t):
    assert t.shape == (7,)
    assert t.dtype == tf.int32


def use_held_shape(t):
    assert t.shape == (23,)
    assert t.dtype == tf.float32


def use_tensor_shape(t):
    assert t.shape == (11,)
    assert t.dtype == tf.float32


def use_mixed_mirrored_shape(t):
    assert t.shape == (19,)
    assert t.dtype == tf.float32


def use_mixed_mapped_shape(t):
    assert t.shape == (29,)
    assert t.dtype == tf.float32


def use_mixed_root_dtype(t):
    assert t.shape == (2,)
    assert t.dtype == tf.string


def use_nested_shape(t):
    assert t.shape == (31,)
    assert t.dtype == tf.float32


def use_keyed_shape(t):
    assert t.shape == (37,)
    assert t.dtype == tf.float32


def use_shuffled_shape(t):
    assert t.shape == (41,)
    assert t.dtype == tf.float32


def use_batch_member_size(t):
    assert t.shape == (43, 2)
    assert t.dtype == tf.float32


def use_held_batch_size(t):
    assert t.shape == (47, 2)
    assert t.dtype == tf.float32


def use_padded(t):
    assert t.shape[1:] == (53,)
    assert t.dtype == tf.int32


def use_padded_default(t):
    assert t.shape[1:] == (53,)
    assert t.dtype == tf.int32


def use_applied_row(t):
    assert t.shape[1:] == (59,)
    assert t.dtype == tf.int32


class Holder:
    pass


def pad(dataset):
    shapes = tf.nest.map_structure(lambda spec: spec.shape, dataset.element_spec)
    consume_applied_shapes_member(shapes[0])
    consume_applied_shape_leaf(shapes[0]["ids"])
    return dataset.padded_batch(2, padded_shapes=shapes)


with tempfile.TemporaryDirectory() as directory:
    path = os.path.join(directory, "lines.txt")
    with open(path, "w") as lines:
        lines.write("a\nb\nc\nd\n")
    other_path = os.path.join(directory, "other.txt")
    with open(other_path, "w") as lines:
        lines.write("e\nf\ng\nh\n")

    # A zip of two functions' results over two files' lines, each a dict.
    zipped = tf.data.Dataset.zip(
        (
            tf.data.TextLineDataset(path).map(
                lambda line: {"ids": tf.ones([5], tf.int32), "length": tf.constant(5)}
            ),
            tf.data.TextLineDataset(other_path).map(
                lambda line: {"ids": tf.ones([7], tf.int32), "length": tf.constant(7)}
            ),
        )
    )
    consume_spec(zipped.element_spec)
    consume_member_spec(zipped.element_spec[0])
    consume_leaf_spec(zipped.element_spec[0]["ids"])
    consume_leaf_shape(zipped.element_spec[0]["ids"].shape)
    use_leaf_shape(tf.zeros(zipped.element_spec[0]["ids"].shape))
    use_other_leaf_shape(tf.zeros(zipped.element_spec[1]["ids"].shape))
    use_leaf_dtype(tf.zeros([2], dtype=zipped.element_spec[0]["ids"].dtype))
    # The shapes a program maps out of the specs.
    shapes = tf.nest.map_structure(lambda spec: spec.shape, zipped.element_spec)
    consume_shapes(shapes)
    consume_shapes_member(shapes[0])
    consume_shape_leaf(shapes[0]["ids"])
    for row in zipped:
        use_row(row[1]["ids"])

    # A spec stored in an attribute and read back.
    held = tf.data.Dataset.zip(
        (
            tf.data.TextLineDataset(path).map(lambda line: {"ids": tf.ones([3])}),
            tf.data.TextLineDataset(other_path).map(
                lambda line: {"ids": tf.ones([23])}
            ),
        )
    )
    holder = Holder()
    holder.spec = held.element_spec[1]["ids"]
    use_held_shape(tf.zeros(holder.spec.shape))

    # A zip of two functions' results that are each one tensor.
    tensors = tf.data.Dataset.zip(
        (
            tf.data.TextLineDataset(path).map(lambda line: tf.ones([11])),
            tf.data.TextLineDataset(other_path).map(lambda line: tf.ones([13])),
        )
    )
    consume_tensor_spec(tensors.element_spec[0])
    use_tensor_shape(tf.zeros(tensors.element_spec[0].shape))
    tensor_shapes = tf.nest.map_structure(lambda spec: spec.shape, tensors.element_spec)
    consume_tensor_shape(tensor_shapes[0])

    # A zip of a dataset whose element is mirrored, a function's results and a file's lines.
    mixed = tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_tensor_slices({"a": tf.ones([4, 19])}),
            tf.data.TextLineDataset(path).map(lambda line: {"ids": tf.ones([29])}),
            tf.data.TextLineDataset(other_path),
        )
    )
    consume_mixed_mirrored_spec(mixed.element_spec[0])
    consume_mixed_mirrored_leaf_spec(mixed.element_spec[0]["a"])
    consume_mixed_mapped_spec(mixed.element_spec[1])
    consume_mixed_root_spec(mixed.element_spec[2])
    use_mixed_mirrored_shape(tf.zeros(mixed.element_spec[0]["a"].shape))
    use_mixed_mapped_shape(tf.zeros(mixed.element_spec[1]["ids"].shape))
    use_mixed_root_dtype(tf.zeros([2], dtype=mixed.element_spec[2].dtype))
    mixed_shapes = tf.nest.map_structure(lambda spec: spec.shape, mixed.element_spec)
    consume_mixed_shape_leaf(mixed_shapes[1]["ids"])

    # A zip of two files' lines, and a zip of two datasets whose elements are mirrored.
    roots = tf.data.Dataset.zip(
        (tf.data.TextLineDataset(path), tf.data.TextLineDataset(other_path))
    )
    consume_roots_spec(roots.element_spec)
    consume_roots_member_spec(roots.element_spec[0])
    mirrored = tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_tensor_slices({"a": tf.ones([4, 3])}),
            tf.data.Dataset.from_tensor_slices({"b": tf.ones([4, 2])}),
        )
    )
    consume_mirrored_spec(mirrored.element_spec)
    consume_mirrored_leaf_spec(mirrored.element_spec[0]["a"])

    # A zip in a zip.
    nested = tf.data.Dataset.zip(
        (
            tf.data.Dataset.zip(
                (
                    tf.data.TextLineDataset(path).map(lambda line: tf.ones([2])),
                    tf.data.TextLineDataset(other_path).map(
                        lambda line: {"ids": tf.ones([31])}
                    ),
                )
            ),
            tf.data.TextLineDataset(path).map(lambda line: tf.ones([3])),
        )
    )
    consume_nested_leaf_spec(nested.element_spec[0][1]["ids"])
    use_nested_shape(tf.zeros(nested.element_spec[0][1]["ids"].shape))

    # A zip of a dict of datasets.
    keyed = tf.data.Dataset.zip(
        {
            "s": tf.data.TextLineDataset(path).map(lambda line: tf.ones([37])),
            "t": tf.data.TextLineDataset(other_path).map(lambda line: tf.ones([2])),
        }
    )
    consume_keyed_spec(keyed.element_spec)
    consume_keyed_leaf_spec(keyed.element_spec["s"])
    use_keyed_shape(tf.zeros(keyed.element_spec["s"].shape))

    # A dataset made from a zip, which hands the element on.
    shuffled = tf.data.Dataset.zip(
        (
            tf.data.TextLineDataset(path).map(lambda line: {"ids": tf.ones([41])}),
            tf.data.TextLineDataset(other_path).map(lambda line: tf.ones([2])),
        )
    ).shuffle(4)
    consume_shuffled_leaf_spec(shuffled.element_spec[0]["ids"])
    use_shuffled_shape(tf.zeros(shuffled.element_spec[0]["ids"].shape))

    # A batch of a zip: the spec has the batch axis.
    batched = tf.data.Dataset.zip(
        (
            tf.data.TextLineDataset(path).map(lambda line: {"ids": tf.ones([43])}),
            tf.data.TextLineDataset(other_path).map(lambda line: tf.ones([2])),
        )
    ).batch(2)
    assert batched.element_spec[0]["ids"].shape.as_list() == [None, 43]
    consume_batched_leaf_shape(batched.element_spec[0]["ids"].shape)
    use_batch_member_size(tf.zeros([batched.element_spec[0]["ids"].shape[1], 2]))
    wide = tf.data.Dataset.zip(
        (
            tf.data.TextLineDataset(path).map(lambda line: {"ids": tf.ones([47])}),
            tf.data.TextLineDataset(other_path).map(lambda line: tf.ones([2])),
        )
    ).batch(2)
    batch_holder = Holder()
    batch_holder.spec = wide.element_spec[0]["ids"]
    use_held_batch_size(tf.zeros([batch_holder.spec.shape[1], 2]))

    # The shapes a program maps out of the specs to pad a batch with, beside the default padding.
    padded = tf.data.Dataset.zip(
        (
            tf.data.TextLineDataset(path).map(
                lambda line: {"ids": tf.ones([53], tf.int32)}
            ),
            tf.data.TextLineDataset(other_path).map(lambda line: tf.ones([2])),
        )
    )
    padded_shapes = tf.nest.map_structure(lambda spec: spec.shape, padded.element_spec)
    for element in padded.padded_batch(2, padded_shapes=padded_shapes):
        use_padded(element[0]["ids"])
    plain = tf.data.Dataset.zip(
        (
            tf.data.TextLineDataset(path).map(
                lambda line: {"ids": tf.ones([53], tf.int32)}
            ),
            tf.data.TextLineDataset(other_path).map(lambda line: tf.ones([2])),
        )
    )
    for element in plain.padded_batch(2):
        use_padded_default(element[0]["ids"])

    # The same, in a function a lambda handed to `apply` passes its dataset to.
    applied = tf.data.Dataset.zip(
        (
            tf.data.TextLineDataset(path).map(
                lambda line: {"ids": tf.ones([59], tf.int32)}
            ),
            tf.data.TextLineDataset(other_path).map(lambda line: tf.ones([2])),
        )
    ).apply(lambda dataset: pad(dataset))
    for element in applied:
        use_applied_row(element[0]["ids"])
