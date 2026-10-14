# Test that enumerating an attribute that two calls write datasets to types the element as either
# dataset's, read by key and by index.
import tensorflow as tf


def dict_sink(x):
    return x


def tuple_sink(x):
    return x


def expanded_sink(x):
    return x


class DictLoop:
    def set(self, dataset):
        self.loader = dataset

    def run(self):
        for step, element in enumerate(self.loader, 1):
            dict_sink(element["x"])


class TupleLoop:
    def set(self, dataset):
        self.loader = dataset

    def run(self):
        for step, element in enumerate(self.loader, 1):
            tuple_sink(element[0])
            expanded_sink(tf.expand_dims(element[0], -1))


floats = tf.ones([4, 3])
ints = tf.ones([4, 3], dtype=tf.int32)

dict_loop = DictLoop()
dict_loop.set(tf.data.Dataset.from_tensor_slices({"x": floats}).batch(2))
dict_loop.run()
dict_loop.set(tf.data.Dataset.from_tensor_slices({"x": ints}).batch(2))
dict_loop.run()

tuple_loop = TupleLoop()
tuple_loop.set(tf.data.Dataset.from_tensor_slices((floats, floats)).batch(2))
tuple_loop.run()
tuple_loop.set(tf.data.Dataset.from_tensor_slices((ints, floats)).batch(2))
tuple_loop.run()
