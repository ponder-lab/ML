# A tensor shape reaches a program through more than the `shape` attribute (wala/ML#1280): the
# `get_shape` method, a dataset's `element_spec`, `tf.compat.v1.data.get_output_shapes`, and a
# `tf.nest.map_structure` over a structure of specs. Each sink below receives one.
import json

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

named = tf.data.Dataset.from_tensor_slices({"a": x, "b": tf.zeros([4])}).batch(2)
consume_dict_spec_shape(named.element_spec["a"].shape)

use_spec_shape(tf.zeros([2] + dataset.element_spec.shape.as_list()[1:]))

# A control: the `shape` attribute of an element the program iterates.
for element in dataset:
    consume_element_shape(element.shape)
    break
