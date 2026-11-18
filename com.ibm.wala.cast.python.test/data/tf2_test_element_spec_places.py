# Test for https://github.com/wala/ML/issues/1300: the `element_spec` of a `map` over a file's lines
# that several functions reach. The analysis describes what the functions return once per place in
# the element, whatever number of functions return a value there, and a spec reads the shape and
# dtype of the dataset's element at its place.
import collections
import os
import tempfile

import tensorflow as tf

Pair = collections.namedtuple("Pair", ["left", "right"])


def parse_first(line):
    return {"ids": tf.ones([4, 2]), "mask": tf.zeros([4], tf.int32)}


def parse_second(line):
    return {"ids": tf.ones([4, 2]), "mask": tf.zeros([4], tf.int32)}


def parse_third(line):
    return {"ids": tf.ones([4, 2]), "mask": tf.zeros([4], tf.int32)}


def flat(line):
    return {"x": tf.ones([5])}


def split(line):
    return {"x": (tf.ones([5]), tf.zeros([3]))}


def pair_first(line):
    return Pair(left=tf.ones([6]), right=tf.zeros([7], tf.int32))


def pair_second(line):
    return Pair(left=tf.ones([6]), right=tf.zeros([7], tf.int32))


def nest_first(line):
    return {"o": (tf.ones([8]), {"i": tf.zeros([9, 2], tf.int32)})}


def nest_second(line):
    return {"o": (tf.ones([8]), {"i": tf.zeros([9, 2], tf.int32)})}


def deep_first(line):
    return {
        "a": {"a": {"a": {"a": {"a": {"a": {"a": {"a": {"a": tf.ones([2, 7])}}}}}}}}
    }


def deep_second(line):
    return {
        "a": {"a": {"a": {"a": {"a": {"a": {"a": {"a": {"a": tf.ones([2, 7])}}}}}}}}
    }


def alone(line):
    return {"k": tf.ones([3, 5])}


# Each dataset below is made by a function of its own, called from a loop over the functions to
# map: one `map` call then makes one dataset, whose element is what any of the functions returns.


def make_several(path, parse):
    return tf.data.TextLineDataset(path).map(parse)


def make_varied(path, parse):
    return tf.data.TextLineDataset(path).map(parse)


def make_paired(path, parse):
    return tf.data.TextLineDataset(path).map(parse)


def make_nested(path, parse):
    return tf.data.TextLineDataset(path).map(parse)


def make_deep(path, parse):
    return tf.data.TextLineDataset(path).map(parse)


def consume_several_spec(spec):
    pass


def consume_several_leaf_spec(spec):
    pass


def consume_several_shapes(shapes):
    pass


def consume_several_leaf_shape(shape):
    pass


def consume_varied_spec(spec):
    pass


def consume_varied_member_spec(spec):
    pass


def consume_paired_spec(spec):
    pass


def consume_paired_left_spec(spec):
    pass


def consume_paired_left_shape(shape):
    pass


def consume_nested_leaf_spec(spec):
    pass


def consume_deep_inner_spec(spec):
    pass


def consume_deep_leaf_spec(spec):
    pass


def consume_alone_spec(spec):
    pass


def consume_alone_leaf_spec(spec):
    pass


def consume_either_spec(spec):
    pass


def use_several_shape(t):
    assert t.shape == (4, 2)
    assert t.dtype == tf.float32


def use_several_dtype(t):
    assert t.shape == (2,)
    assert t.dtype == tf.int32


def use_paired_shape(t):
    assert t.shape == (6,)
    assert t.dtype == tf.float32


def use_paired_dtype(t):
    assert t.shape == (2,)
    assert t.dtype == tf.int32


def use_nested_shape(t):
    assert t.shape == (9, 2)
    assert t.dtype == tf.float32


def use_deep_shape(t):
    assert t.shape == (2, 7)
    assert t.dtype == tf.float32


def use_alone_shape(t):
    assert t.shape == (3, 5)
    assert t.dtype == tf.float32


with tempfile.TemporaryDirectory() as directory:
    path = os.path.join(directory, "lines.txt")
    with open(path, "w") as lines:
        lines.write("a\nb\nc\nd\n")

    # Three functions that return dicts with the same keys.
    for parse in [parse_first, parse_second, parse_third]:
        several = make_several(path, parse)
    consume_several_spec(several.element_spec)
    consume_several_leaf_spec(several.element_spec["ids"])
    use_several_shape(tf.zeros(several.element_spec["ids"].shape))
    use_several_dtype(tf.zeros([2], dtype=several.element_spec["mask"].dtype))

    # The shapes a program maps out of those specs.
    shapes = tf.nest.map_structure(lambda spec: spec.shape, several.element_spec)
    consume_several_shapes(shapes)
    consume_several_leaf_shape(shapes["ids"])

    # Two functions whose dicts hold a tensor and a tuple under one key: the key's place holds a
    # spec and a tuple of specs.
    for parse in [flat, split]:
        varied = make_varied(path, parse)
    consume_varied_spec(varied.element_spec)
    consume_varied_member_spec(varied.element_spec["x"])

    # Two functions that return a namedtuple instance, read by a field's name, also after the
    # shapes are mapped out of the specs.
    for parse in [pair_first, pair_second]:
        paired = make_paired(path, parse)
    consume_paired_spec(paired.element_spec)
    consume_paired_left_spec(paired.element_spec.left)
    use_paired_shape(tf.zeros(paired.element_spec.left.shape))
    use_paired_dtype(tf.zeros([2], dtype=paired.element_spec.right.dtype))
    paired_shapes = tf.nest.map_structure(lambda spec: spec.shape, paired.element_spec)
    consume_paired_left_shape(paired_shapes.left)

    # Two functions that return a dict in a tuple in a dict.
    for parse in [nest_first, nest_second]:
        nested = make_nested(path, parse)
    consume_nested_leaf_spec(nested.element_spec["o"][1]["i"])
    use_nested_shape(tf.zeros(nested.element_spec["o"][1]["i"].shape))

    # Two functions that return a dict nine levels deep. The analysis records the place of a value
    # down to eight levels, so the innermost dict has a place and its value has none.
    for parse in [deep_first, deep_second]:
        deep = make_deep(path, parse)
    inner = deep.element_spec["a"]["a"]["a"]["a"]["a"]["a"]["a"]["a"]
    consume_deep_inner_spec(inner)
    consume_deep_leaf_spec(inner["a"])
    use_deep_shape(tf.zeros(inner["a"].shape))

    # Controls. One function: one description of the dict and of its value.
    single = tf.data.TextLineDataset(path).map(alone)
    consume_alone_spec(single.element_spec)
    consume_alone_leaf_spec(single.element_spec["k"])
    use_alone_shape(tf.zeros(single.element_spec["k"].shape))

    # Two datasets read at one place in the program: each has a description of its own.
    one = tf.data.TextLineDataset(path).map(lambda line: {"k": tf.ones([3])})
    other = tf.data.TextLineDataset(path).map(lambda line: {"k": tf.ones([3])})
    for dataset in [one, other]:
        consume_either_spec(dataset.element_spec)
