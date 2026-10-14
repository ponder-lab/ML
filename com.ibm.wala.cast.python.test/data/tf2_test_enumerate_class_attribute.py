# Test that enumerating an attribute the class body assigns types the element, alone and beside an
# instance write of the same attribute.
import tensorflow as tf


def class_sink(x):
    return x


def mixed_sink(x):
    return x


def make_floats():
    return tf.data.Dataset.from_tensor_slices({"x": tf.ones([4, 3])}).batch(2)


def make_ints():
    return tf.data.Dataset.from_tensor_slices(
        {"x": tf.ones([4, 3], dtype=tf.int32)}
    ).batch(2)


class ClassLoop:
    loader = make_floats()

    def run(self):
        for step, batch in enumerate(self.loader, 1):
            class_sink(batch["x"])


class MixedLoop:
    loader = make_floats()

    def replace(self):
        self.loader = make_ints()

    def run(self):
        for step, batch in enumerate(self.loader, 1):
            mixed_sink(batch["x"])


ClassLoop().run()

mixed = MixedLoop()
mixed.run()
mixed.replace()
mixed.run()
