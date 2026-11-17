# The `element_spec` of a dataset whose element is not mirrored: a dataset of a file's lines or
# records or of file names, a `map` over one, and a dataset made from either (wala/ML#1280). The
# spec describes the element of the dataset it is read off, so a batch's spec has the batch axis.

import os
import tempfile

import tensorflow as tf


def consume_spec(spec):
    pass


def consume_member_spec(spec):
    pass


def use_line_shape(t):
    assert t.shape == ()
    assert t.dtype == tf.float32


def use_line_dtype(t):
    assert t.shape == (2,)
    assert t.dtype == tf.string


def use_shuffled_shape(t):
    assert t.shape == ()
    assert t.dtype == tf.float32


def use_file_name_dtype(t):
    assert t.shape == (2,)
    assert t.dtype == tf.string


def use_record_dtype(t):
    assert t.shape == (2,)
    assert t.dtype == tf.string


def use_member_shape(t):
    assert t.shape == (3,)
    assert t.dtype == tf.float32


def use_member_dtype(t):
    assert t.shape == (2,)
    assert t.dtype == tf.int32


def use_batch_member_size(t):
    assert t.shape == (3, 2)
    assert t.dtype == tf.float32


def use_batch_member_sizes(t):
    assert t.shape == (2, 2)
    assert t.dtype == tf.float32


def use_batch_axis(t):
    # Not called at run time, where the batch axis has no size.
    assert t.shape == (2, 5)
    assert t.dtype == tf.float32


def use_tensor_shape(t):
    assert t.shape == (5,)
    assert t.dtype == tf.float32


def use_tensor_batch_size(t):
    assert t.shape == (5,)
    assert t.dtype == tf.float32


def use_pair_shape(t):
    assert t.shape == (7,)
    assert t.dtype == tf.float32


def use_pair_dtype(t):
    assert t.shape == (2,)
    assert t.dtype == tf.int32


def use_nested_shape(t):
    assert t.shape == (9, 2)
    assert t.dtype == tf.float32


def use_generated_shape(t):
    assert t.shape == (4,)
    assert t.dtype == tf.float32


def use_generated_dtype(t):
    assert t.shape == (2,)
    assert t.dtype == tf.int32


def use_passed_shape(t):
    assert t.shape == (6,)
    assert t.dtype == tf.float32


def use_padded(t):
    assert t.shape == (2, 3)
    assert t.dtype == tf.float32


def use_padded_default(t):
    assert t.shape == (2, 3)
    assert t.dtype == tf.float32


def use_held_shape(t):
    assert t.shape == (4,)
    assert t.dtype == tf.float32


def use_held_batch_size(t):
    assert t.shape == (11, 2)
    assert t.dtype == tf.float32


def use_flat_shape(t):
    assert t.shape == (9,)
    assert t.dtype == tf.float32


class Holder:
    pass


def zeros_of(spec):
    return tf.zeros(spec.shape)


def generate():
    yield tf.ones([4])


def generate_ints():
    yield tf.ones([3], tf.int32)


with tempfile.TemporaryDirectory() as directory:
    path = os.path.join(directory, "lines.txt")
    with open(path, "w") as lines:
        lines.write("a\nb\nc\nd\n")

    # A dataset of lines: the element is one value, a scalar string.
    text = tf.data.TextLineDataset(path)
    use_line_shape(tf.zeros(text.element_spec.shape))
    use_line_dtype(tf.zeros([2], dtype=text.element_spec.dtype))

    # A dataset made from one, which hands the element on.
    shuffled = tf.data.TextLineDataset(path).shuffle(4)
    use_shuffled_shape(tf.zeros(shuffled.element_spec.shape))

    # A dataset of file names and a dataset of records.
    names = tf.data.Dataset.list_files(os.path.join(directory, "*.txt"))
    use_file_name_dtype(tf.zeros([2], dtype=names.element_spec.dtype))
    records = tf.data.TFRecordDataset(path)
    use_record_dtype(tf.zeros([2], dtype=records.element_spec.dtype))

    # A function's results over the lines: a dict whose values differ in shape and dtype.
    mapped = tf.data.TextLineDataset(path).map(
        lambda line: {"a": tf.ones([3]), "b": tf.zeros([2, 2], tf.int32)}
    )
    consume_spec(mapped.element_spec)
    consume_member_spec(mapped.element_spec["a"])
    use_member_shape(tf.zeros(mapped.element_spec["a"].shape))
    use_member_dtype(tf.zeros([2], dtype=mapped.element_spec["b"].dtype))

    # A batch of them: the same function's results, with the batch axis.
    batched = mapped.batch(2)
    a = batched.element_spec["a"]
    b = batched.element_spec["b"]
    assert a.shape.as_list() == [None, 3]
    assert b.shape.as_list() == [None, 2, 2]
    # The batch axis has no size, since the last batch may be smaller.
    if a.shape[0] is not None:
        use_batch_axis(tf.zeros([a.shape[0], 5]))
    use_batch_member_size(tf.zeros([a.shape[1], 2]))
    use_batch_member_sizes(tf.zeros([b.shape[1], b.shape[2]]))

    # A function that returns one tensor, and a batch of its results.
    tensors = tf.data.TextLineDataset(path).map(lambda line: tf.ones([5]))
    use_tensor_shape(tf.zeros(tensors.element_spec.shape))
    use_tensor_batch_size(tf.zeros([tensors.batch(2).element_spec.shape[1]]))

    # A function that returns a tuple, over records.
    pair = tf.data.TFRecordDataset(path).map(
        lambda record: (tf.ones([6]), tf.zeros([7], tf.int32))
    )
    use_pair_shape(tf.zeros(pair.element_spec[1].shape))
    use_pair_dtype(tf.zeros([2], dtype=pair.element_spec[1].dtype))
    # A spec passed to a function.
    use_passed_shape(zeros_of(pair.element_spec[0]))

    # A spec stored in an attribute and read back, of a `map`'s results and of a batch of them.
    held = tf.data.TextLineDataset(path).map(
        lambda line: {"a": tf.ones([4]), "b": tf.zeros([2, 2], tf.int32)}
    )
    holder = Holder()
    holder.spec = held.element_spec["a"]
    use_held_shape(tf.zeros(holder.spec.shape))
    wide = tf.data.TextLineDataset(path).map(
        lambda line: {"a": tf.ones([11]), "b": tf.zeros([2, 2], tf.int32)}
    )
    batch_holder = Holder()
    batch_holder.spec = wide.batch(2).element_spec["a"]
    use_held_batch_size(tf.zeros([batch_holder.spec.shape[1], 2]))

    # A spec read out of the specs' leaves.
    single = tf.data.TextLineDataset(path).map(lambda line: {"a": tf.ones([9])})
    use_flat_shape(tf.zeros(tf.nest.flatten(single.element_spec)[0].shape))

    # A dict in a tuple in a dict.
    nested = tf.data.TextLineDataset(path).map(
        lambda line: {"o": (tf.ones([8]), {"i": tf.zeros([9, 2], tf.int32)})}
    )
    use_nested_shape(tf.zeros(nested.element_spec["o"][1]["i"].shape))

    # The shapes a program maps out of the specs to pad a batch with, beside the default padding.
    padded = tf.data.TextLineDataset(path).map(
        lambda line: {"a": tf.ones([3]), "b": tf.zeros([2, 2], tf.int32)}
    )
    shapes = tf.nest.map_structure(lambda spec: spec.shape, padded.element_spec)
    for element in padded.padded_batch(2, padded_shapes=shapes):
        use_padded(element["a"])
    plain = tf.data.TextLineDataset(path).map(
        lambda line: {"a": tf.ones([3]), "b": tf.zeros([2, 2], tf.int32)}
    )
    for element in plain.padded_batch(2):
        use_padded_default(element["a"])

# A dataset of a generator's values, which declares its element.
generated = tf.data.Dataset.from_generator(
    generate, output_signature=tf.TensorSpec([4], tf.float32)
)
use_generated_shape(tf.zeros(generated.element_spec.shape))
typed = tf.data.Dataset.from_generator(generate_ints, output_types=tf.int32)
use_generated_dtype(tf.zeros([2], dtype=typed.element_spec.dtype))
