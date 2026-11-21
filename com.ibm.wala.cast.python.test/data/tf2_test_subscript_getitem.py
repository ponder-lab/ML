# Test that a subscript of an object whose class defines `__getitem__` calls the method (wala/ML#1306).
import numpy as np
import tensorflow as tf


def in_body(x):
    assert x.shape == (2, 4) and x.dtype == np.float32


def by_subscript(x):
    assert x.shape == (2, 4) and x.dtype == np.float32


def by_method(x):
    assert x.shape == (2, 4) and x.dtype == np.float32


def by_explicit_call(x):
    assert x.shape == (2, 4) and x.dtype == np.float32


def by_unpacking(x):
    assert x.shape == (2, 4) and x.dtype == np.float32


def by_unpacking_other(x):
    assert x.shape == (2,) and x.dtype == np.int64


def variable_of_subscript(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def by_inherited(x):
    assert x.shape == (2, 4) and x.dtype == np.float32


def per_instance(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def per_instance_other(x):
    assert x.shape == (2,) and x.dtype == tf.int32


def by_index(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def by_index_other(x):
    assert x.shape == (2,) and x.dtype == tf.int32


def by_string_key(x):
    assert x.shape == (2,) and x.dtype == tf.int32


def key_in_body(x):
    assert x.shape == (2,) and x.dtype == tf.int32


def in_function(x):
    assert x.shape == (2, 4) and x.dtype == np.float32


def by_wrapper(x):
    assert x.shape == (2, 4) and x.dtype == np.float32


def by_list(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def by_index_method(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def by_index_method_other(x):
    assert x.shape == (2,) and x.dtype == tf.int32


def in_method(x):
    assert x.shape == (2, 4) and x.dtype == np.float32


def in_loop(x):
    assert x.shape == (2, 4) and x.dtype == np.float32


def in_comprehension(x):
    assert x.shape == (2, 4) and x.dtype == np.float32


def of_sequence(x):
    assert x.shape == (2, 4) and x.dtype == np.float32


def of_either(x):
    assert (x.shape == (2, 4) and x.dtype == np.float32) or (
        x.shape == (3,) and x.dtype == tf.int32
    )


def by_slice(x):
    assert x.shape == (2, 4) and x.dtype == np.float32


def by_tuple_key(x):
    assert x.shape == (2, 4) and x.dtype == np.float32


def by_iteration(x):
    assert x.shape == (2, 4) and x.dtype == np.float32


def by_attribute_name(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def by_dict_subclass(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def by_constant_receiver(x):
    assert x == "b"


def of_two_classes(x):
    assert x.shape == (2, 4) and x.dtype in (np.float32, tf.int64)


class Boxes:
    def __getitem__(self, index):
        boxes = np.array([[1, 1, 4, 4], [2, 2, 6, 6]], dtype=np.float32)
        in_body(boxes)
        return boxes

    def get(self, index):
        return np.array([[1, 1, 4, 4], [2, 2, 6, 6]], dtype=np.float32)


class Called:
    def __getitem__(self, index):
        return np.array([[1, 1, 4, 4], [2, 2, 6, 6]], dtype=np.float32)


class Pairs:
    def __getitem__(self, index):
        boxes = np.array([[1, 1, 4, 4], [2, 2, 6, 6]], dtype=np.float32)
        labels = np.array([1, 2], dtype=np.int64)
        return boxes, labels


class Base:
    def __getitem__(self, index):
        return np.array([[1, 1, 4, 4], [2, 2, 6, 6]], dtype=np.float32)


class Derived(Base):
    pass


class Holder:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, index):
        return self.value


class Indexed:
    def __init__(self):
        self.items = (tf.ones((2, 4)), tf.ones((2,), tf.int32))

    def __getitem__(self, index):
        return self.items[index]

    def get(self, index):
        return self.items[index]


class Named:
    def __init__(self):
        self.entries = {"boxes": tf.ones((2, 4)), "labels": tf.ones((2,), tf.int32)}

    def __getitem__(self, key):
        return self.entries[key]


class Keyed:
    def __getitem__(self, key):
        key_in_body(key)
        return 0


class Wrapper:
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def __getitem__(self, index):
        return self.wrapped[index]


class Reader:
    def __getitem__(self, index):
        return np.array([[1, 1, 4, 4], [2, 2, 6, 6]], dtype=np.float32)

    def first(self):
        return self[0]


class Batches(tf.keras.utils.Sequence):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return np.array([[1, 1, 4, 4], [2, 2, 6, 6]], dtype=np.float32)


class Finite:
    def __getitem__(self, index):
        if index >= 2:
            raise IndexError(index)
        return np.array([[1, 1, 4, 4], [2, 2, 6, 6]], dtype=np.float32)


class Tagged:
    def __init__(self):
        self.labels = tf.ones((3,), tf.int32)

    def __getitem__(self, key):
        return tf.ones((2, 4))


class Table(dict):
    def __getitem__(self, key):
        return super().__getitem__(key)


class Marks:
    def __getitem__(self, index):
        return tf.ones((2, 4), tf.int64)


def either(flag):
    if flag:
        return Boxes()
    return Marks()


def read(holder):
    in_function(holder[0])


def read_either(holder):
    of_either(holder[0])


boxes = Boxes()
by_subscript(boxes[0])
by_method(boxes.get(0))
by_explicit_call(Called().__getitem__(0))
variable_of_subscript(tf.Variable(boxes[0]))
first, second = Pairs()[0]
by_unpacking(first)
by_unpacking_other(second)
by_inherited(Derived()[0])
first_holder = Holder(tf.ones((2, 4)))
second_holder = Holder(tf.ones((2,), tf.int32))
per_instance(first_holder[0])
per_instance_other(second_holder[0])
indexed = Indexed()
by_index(indexed[0])
by_index_other(indexed[1])
by_string_key(Named()["labels"])
Keyed()[tf.constant([1, 2])]
read(Boxes())
by_wrapper(Wrapper(Boxes())[0])
by_list([tf.ones((2, 4))][0])
by_index_method(indexed.get(0))
by_index_method_other(indexed.get(1))
in_method(Reader().first())
for i in range(2):
    in_loop(boxes[i])
rows = [boxes[i] for i in range(2)]
in_comprehension(rows[0])
of_sequence(Batches()[0])
read_either(Boxes())
read_either([tf.ones((3,), tf.int32)])
by_slice(boxes[0:1])
by_tuple_key(boxes[0, 1])
for row in Finite():
    by_iteration(row)
by_attribute_name(Tagged()["labels"])
table = Table()
table["boxes"] = tf.ones((2, 4))
by_dict_subclass(table["boxes"])
by_constant_receiver("boxes"[0])
for flag in (True, False):
    of_two_classes(either(flag)[0])


class Config(dict):
    def describe(self):
        return len(self)


def by_plain_dict_subclass(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def read_plain(config):
    return config["boxes"]


config = Config()
config["boxes"] = tf.ones((2, 4))
assert config.describe() == 1
by_plain_dict_subclass(read_plain(config))

SEEN = []


class Noted(tf.keras.layers.Layer):
    def call(self, x, training=None):
        SEEN.append(training)
        return x


class Applies:
    def __init__(self):
        self.layer = Noted()

    def __getitem__(self, x):
        return self.layer(x)


class Subscripts(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.applies = Applies()

    def call(self, x, training=None):
        return self.applies[x]


Subscripts()(tf.ones((2, 3)), training=True)
assert SEEN == [True], SEEN
