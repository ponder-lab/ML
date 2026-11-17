# A tensor shape reaches a program through more than the `shape` attribute (wala/ML#1280): the
# `get_shape` method, a dataset's `element_spec`, `tf.compat.v1.data.get_output_shapes`, and a
# `tf.nest.map_structure` over a structure of specs. Each sink below receives one.
import collections
import json
import os
import tempfile

import tensorflow as tf


def consume_attribute(s):
    assert isinstance(s, tf.TensorShape)
    assert s.as_list() == [4, 3]


def consume_method(s):
    assert isinstance(s, tf.TensorShape)
    assert s.as_list() == [4, 3]


def consume_method_list(sizes):
    assert sizes == [4, 3]


def consume_variable_method(s):
    assert isinstance(s, tf.TensorShape)
    assert s.as_list() == [2]


def consume_spec(spec):
    assert isinstance(spec, tf.TensorSpec)


def consume_spec_shape(s):
    assert isinstance(s, tf.TensorShape)
    assert s.as_list() == [None, 3]


def consume_tuple_spec_shape(s):
    assert isinstance(s, tf.TensorShape)
    assert s.as_list() == [None, 3]


def consume_dict_spec_shape(s):
    assert isinstance(s, tf.TensorShape)
    assert s.as_list() == [None, 3]


def consume_tuple_spec(specs):
    assert isinstance(specs, tuple)
    assert isinstance(specs[0], tf.TensorSpec)


def consume_tuple_spec_member(spec):
    assert isinstance(spec, tf.TensorSpec)


def consume_nested_spec_shape(s):
    assert isinstance(s, tf.TensorShape)
    assert s.as_list() == [None, 3]


def consume_named_spec_shape(s):
    assert isinstance(s, tf.TensorShape)
    assert s.as_list() == [None, 3]


def consume_optional_spec(spec):
    assert isinstance(spec, tf.TensorSpec)


def spec_of_optional(given):
    chosen = None
    if given:
        chosen = dataset
    if chosen is not None:
        consume_optional_spec(chosen.element_spec)


def consume_parameter_spec_shape(s):
    assert isinstance(s, tf.TensorShape)
    assert s.as_list() == [None, 3]


def consume_map_spec(spec):
    assert isinstance(spec, tf.TensorSpec)


def consume_map_spec_shape(s):
    assert isinstance(s, tf.TensorShape)
    assert s.as_list() == [None, 3]


def consume_constructed_spec_shape(s):
    assert isinstance(s, tf.TensorShape)
    assert s.as_list() == [None, 3]


def consume_text_spec(spec):
    assert isinstance(spec, tf.TensorSpec)


def consume_text_spec_shape(s):
    assert isinstance(s, tf.TensorShape)
    assert s.as_list() == []


def consume_text_map_spec_shape(s):
    assert isinstance(s, tf.TensorShape)
    assert s.as_list() == [3]


def consume_text_mapped(s):
    assert isinstance(s, tf.TensorShape)
    assert s.as_list() == [3]


def consume_text_dict_spec(spec):
    assert isinstance(spec, dict)


def consume_text_dict_spec_member(spec):
    assert isinstance(spec, tf.TensorSpec)


def consume_text_enumerate_spec(spec):
    assert isinstance(spec, tuple)


def consume_text_dict_mapped(shapes):
    assert isinstance(shapes, dict)


def consume_text_dict_mapped_member(s):
    assert isinstance(s, tf.TensorShape)
    assert s.as_list() == [2, 2]


def consume_spec_batch_size(n):
    assert n is None


def consume_keyed_row(t):
    assert t.shape == (3,)


def keyed_row(d):
    consume_keyed_row(d["element_spec"])
    return d["element_spec"]


def consume_keyed_element(t):
    assert t.shape == (3,)


def use_constructed_spec_shape(t):
    assert t.shape == (2, 3)


def consume_output_shapes(s):
    assert isinstance(s, tf.TensorShape)
    assert s.as_list() == [None, 3]


def consume_tuple_output_shape(s):
    assert isinstance(s, tf.TensorShape)
    assert s.as_list() == [None, 3]


def consume_mapped(s):
    assert isinstance(s, tf.TensorShape)
    assert s.as_list() == [None, 3]


def consume_mapped_member(s):
    assert isinstance(s, tf.TensorShape)
    assert s.as_list() == [None, 3]


def consume_mapped_structure(s):
    assert isinstance(s, tuple)
    assert s[0].as_list() == [None, 3]


def consume_parameter_mapped(s):
    assert isinstance(s, tf.TensorShape)
    assert s.as_list() == [None, 3]


def consume_element_shape(s):
    assert isinstance(s, tf.TensorShape)
    assert s.as_list() == [2, 3]


def use_reshape_attribute(t):
    assert t.shape == (4, 3)


def use_reshape_method(t):
    assert t.shape == (4, 3)


def use_zeros_attribute(t):
    assert t.shape == (4, 3)


def use_zeros_method(t):
    assert t.shape == (4, 3)


def use_zeros_attribute_list(t):
    assert t.shape == (4, 3)


def use_zeros_method_list(t):
    assert t.shape == (4, 3)


def use_size_attribute(t):
    assert t.shape == (4, 2)


def use_size_method(t):
    assert t.shape == (4, 2)


def use_list_size_attribute(t):
    assert t.shape == (3, 2)


def use_list_size_method(t):
    assert t.shape == (3, 2)


def use_dimension_attribute(n):
    assert n == 4


def use_dimension_method(n):
    assert n == 4


def use_spec_shape(t):
    assert t.shape == (2, 3)


def use_spec_size(t):
    assert t.shape == (3, 2)


def use_spec_list_size(t):
    assert t.shape == (3, 2)


def use_unbatched_spec_shape(t):
    assert t.shape == (3,)


def use_sliced_list(t):
    assert t.shape == (2, 3)


def use_sliced_list_reshape(t):
    assert t.shape == (2, 3)


def use_spec_reshape(t):
    assert t.shape == (2, 3)


def use_passed_spec_shape(t):
    assert t.shape == (3,)


def zeros_of(spec):
    return tf.zeros(spec.shape)


def use_passed_spec_dtype(t):
    assert t.shape == (2,)
    assert t.dtype == tf.int32


def typed_as(spec):
    return tf.zeros([2], dtype=spec.dtype)


def use_mapped_spec_shape(t):
    assert t.shape == (3,)


def zeros_of_leaf(spec):
    use_mapped_spec_shape(tf.zeros(spec.shape))


def use_mapped_spec_dtype(t):
    assert t.shape == (2,)
    assert t.dtype == tf.int32


def typed_as_leaf(spec):
    use_mapped_spec_dtype(tf.zeros([2], dtype=spec.dtype))


def use_member_spec_shape(t):
    assert t.shape == (3,)
    assert t.dtype == tf.float32


def use_member_spec_dtype(t):
    assert t.shape == (2,)
    assert t.dtype == tf.int32


def use_spec_dtype(t):
    assert t.shape == (2,)
    assert t.dtype == tf.int32


def use_tensor_dtype(t):
    assert t.shape == (2,)
    assert t.dtype == tf.int32


def use_rank_attribute(t):
    assert t.shape == (5,)


def use_rank_method(t):
    assert t.shape == (5,)


def use_parameter_attribute(t):
    assert t.shape == (4, 3)


def use_parameter_method(t):
    assert t.shape == (4, 3)


def use_assigned_list_attribute(t):
    assert t.shape == (4, 2)


def use_assigned_list_method(t):
    assert t.shape == (4, 2)


def use_own_method(t):
    assert t.shape == (7, 7)


def use_either_method(t):
    assert t.shape in ((4, 3), (7, 7))


def use_either_attribute(t):
    assert t.shape in ((4, 3), (7, 7))


def use_opaque_method(t):
    assert t.shape in ((4, 3), (7, 7))


def by_rank_attribute(t):
    if t.shape.ndims == 2:
        return tf.ones([5])
    return tf.ones([6, 6])


def by_rank_method(t):
    if t.get_shape().ndims == 2:
        return tf.ones([5])
    return tf.ones([6, 6])


def zeros_of_attribute(t):
    return tf.zeros(t.shape)


def zeros_of_method(t):
    return tf.zeros(t.get_shape())


class Holder:
    """A program's own class with an attribute named as the tensor's is."""

    def __init__(self):
        self.shape = [7, 7]


class Opaque:
    """A program's own class whose method returns a shape that is not read."""

    def get_shape(self):
        return json.loads("[7, 7]")


def spec_shape_of(d):
    return d.element_spec.shape


def mapped_shapes_of(d):
    return tf.nest.map_structure(lambda spec: spec.shape, d.element_spec)


Pair = collections.namedtuple("Pair", ["a", "b"])


def ones_of_assigned_attribute(t):
    sizes = t.shape.as_list()
    return tf.ones([sizes[0], 2])


def ones_of_assigned_method(t):
    sizes = t.get_shape().as_list()
    return tf.ones([sizes[0], 2])


class Box:
    """A program's own class with a method named as the tensor's is."""

    def __init__(self, t):
        self.t = t

    def get_shape(self):
        return [7, 7]


x = tf.ones([4, 3])
consume_attribute((x * 2).shape)
consume_method((x * 2).get_shape())
consume_method_list(x.get_shape().as_list())
v = tf.Variable(tf.zeros([2]))
consume_variable_method(v.get_shape())

# The forms a program consumes a shape in, each beside the same form over the `shape` attribute.
# A shape passed whole as a list, by `as_list()`, has an unknown rank in either form
# (wala/ML#1285).
y = tf.ones([3, 4])
use_reshape_attribute(tf.reshape(y, x.shape))
use_reshape_method(tf.reshape(y, x.get_shape()))
use_zeros_attribute(tf.zeros(x.shape))
use_zeros_method(tf.zeros(x.get_shape()))
use_zeros_attribute_list(tf.zeros(x.shape.as_list()))
use_zeros_method_list(tf.zeros(x.get_shape().as_list()))
use_size_attribute(tf.ones([x.shape[0], 2]))
use_size_method(tf.ones([x.get_shape()[0], 2]))
use_list_size_attribute(tf.ones([x.shape.as_list()[1], 2]))
use_list_size_method(tf.ones([x.get_shape().as_list()[1], 2]))
use_dimension_attribute(x.shape[0])
use_dimension_method(x.get_shape()[0])
use_rank_attribute(by_rank_attribute(x))
use_rank_method(by_rank_method(x))
use_parameter_attribute(zeros_of_attribute(x))
use_parameter_method(zeros_of_method(x))
use_assigned_list_attribute(ones_of_assigned_attribute(x))
use_assigned_list_method(ones_of_assigned_method(x))
# A control: the method of a program's own class is not the tensor's.
use_own_method(tf.zeros(Box(x).get_shape()))
# A receiver that is the tensor at one iteration and the program's instance at the next, so the
# call has the tensor's method and the program's as targets. The allocator reads the program's
# list alone (wala/ML#1286).
for either_method in [x, Box(x)]:
    use_either_method(tf.zeros(either_method.get_shape()))
for either_attribute in [x, Holder()]:
    use_either_attribute(tf.zeros(either_attribute.shape))
# A control: the same, where what the program's method returns is not read.
for either_opaque in [x, Opaque()]:
    use_opaque_method(tf.zeros(either_opaque.get_shape()))

dataset = tf.data.Dataset.from_tensor_slices(x).batch(2)
consume_spec(dataset.element_spec)
consume_spec_shape(dataset.element_spec.shape)
consume_output_shapes(tf.compat.v1.data.get_output_shapes(dataset))
consume_mapped(tf.nest.map_structure(lambda spec: spec.shape, dataset.element_spec))

pairs = tf.data.Dataset.from_tensor_slices((x, tf.zeros([4]))).batch(2)
consume_tuple_spec_shape(pairs.element_spec[0].shape)
consume_tuple_output_shape(tf.compat.v1.data.get_output_shapes(pairs)[0])
consume_mapped_member(
    tf.nest.map_structure(lambda spec: spec.shape, pairs.element_spec)[0]
)
consume_mapped_structure(
    tf.nest.map_structure(lambda spec: spec.shape, pairs.element_spec)
)

named = tf.data.Dataset.from_tensor_slices({"a": x, "b": tf.zeros([4])}).batch(2)
consume_dict_spec_shape(named.element_spec["a"].shape)

use_spec_shape(tf.zeros([2] + dataset.element_spec.shape.as_list()[1:]))
# A control: the same list built from a tensor's shape.
use_sliced_list(tf.zeros([2] + x.shape.as_list()[1:]))
z = tf.ones([3, 2])
use_sliced_list_reshape(tf.reshape(z, [2] + x.shape.as_list()[1:]))
use_spec_reshape(tf.reshape(z, [2] + dataset.element_spec.shape.as_list()[1:]))
use_spec_size(tf.ones([dataset.element_spec.shape[1], 2]))
use_spec_list_size(tf.ones([dataset.element_spec.shape.as_list()[1], 2]))
# A batch's size is not in its spec, since the last batch may be smaller.
consume_spec_batch_size(dataset.element_spec.shape[0])
plain = tf.data.Dataset.from_tensor_slices(x)
use_unbatched_spec_shape(tf.zeros(plain.element_spec.shape))

# A structure of specs is no spec, and its member is one.
consume_tuple_spec(pairs.element_spec)
consume_tuple_spec_member(pairs.element_spec[0])
nested = tf.data.Dataset.from_tensor_slices(({"a": x}, tf.zeros([4]))).batch(2)
consume_nested_spec_shape(nested.element_spec[0]["a"].shape)
tupled = tf.data.Dataset.from_tensor_slices(Pair(x, tf.zeros([4]))).batch(2)
consume_named_spec_shape(tupled.element_spec.a.shape)
consume_parameter_spec_shape(spec_shape_of(dataset))
consume_parameter_mapped(mapped_shapes_of(dataset))
consume_map_spec_shape(dataset.map(lambda t: t * 2).element_spec.shape)
consume_map_spec(dataset.map(lambda t: t * 3).element_spec)
spec_of_optional(True)

# The spec's dtype, beside a control: a tensor's.
ints = tf.data.Dataset.from_tensor_slices(tf.zeros([4, 3], tf.int32))
use_spec_dtype(tf.zeros([2], dtype=ints.element_spec.dtype))
use_tensor_dtype(tf.zeros([2], dtype=tf.zeros([4, 3], tf.int32).dtype))

# A spec passed on, so that its shape and dtype are read where the dataset is out of sight.
use_passed_spec_shape(zeros_of(plain.element_spec))
use_passed_spec_dtype(typed_as(ints.element_spec))

# The specs of a dict's values, which reach a function as the leaves `tf.nest.map_structure` maps.
twins = tf.data.Dataset.from_tensor_slices(
    {"a": tf.zeros([4, 3], tf.int32), "b": tf.ones([4, 3], tf.int32)}
)
tf.nest.map_structure(zeros_of_leaf, twins.element_spec)
tf.nest.map_structure(typed_as_leaf, twins.element_spec)

# The spec of one value of a dict whose values differ in shape and in dtype, passed on.
mixed = tf.data.Dataset.from_tensor_slices({"a": x, "b": tf.zeros([4], tf.int32)})
use_member_spec_shape(zeros_of(mixed.element_spec["a"]))
use_member_spec_dtype(typed_as(mixed.element_spec["b"]))

# A spec the program constructs.
consume_constructed_spec_shape(tf.TensorSpec([None, 3]).shape)
use_constructed_spec_shape(tf.zeros(tf.TensorSpec([2, 3]).shape))

# A dataset of a file's lines, whose element is a scalar string.
with tempfile.TemporaryDirectory() as directory:
    path = os.path.join(directory, "lines.txt")
    with open(path, "w") as lines:
        lines.write("a\nb\n")
    text = tf.data.TextLineDataset(path)
    consume_text_spec(text.element_spec)
    consume_text_spec_shape(text.element_spec.shape)
    # A function's results over the lines.
    rows = text.map(lambda line: tf.zeros([3]))
    consume_text_map_spec_shape(rows.element_spec.shape)
    # The shapes mapped out of its specs.
    consume_text_mapped(
        tf.nest.map_structure(lambda spec: spec.shape, rows.element_spec)
    )
    # A function's results that are a dict, and the shapes mapped out of its specs.
    pairs = text.map(lambda line: {"a": tf.ones([3]), "b": tf.zeros([2, 2], tf.int32)})
    consume_text_dict_spec(pairs.element_spec)
    consume_text_dict_spec_member(pairs.element_spec["a"])
    pair_shapes = tf.nest.map_structure(lambda spec: spec.shape, pairs.element_spec)
    consume_text_dict_mapped(pair_shapes)
    consume_text_dict_mapped_member(pair_shapes["b"])
    # The lines counted.
    consume_text_enumerate_spec(text.enumerate().element_spec)
    # A control: a member named as the attribute is, of an element of such a dataset.
    keyed_rows = text.map(lambda line: {"element_spec": tf.zeros([3])})
    for row in keyed_rows.map(keyed_row):
        break

# A control: an element's own member named as the attribute is.
keyed = tf.data.Dataset.from_tensor_slices({"element_spec": x})
for item in keyed:
    consume_keyed_element(item["element_spec"])
    break

# A control: the `shape` attribute of an element the program iterates.
for element in dataset:
    consume_element_shape(element.shape)
    break
