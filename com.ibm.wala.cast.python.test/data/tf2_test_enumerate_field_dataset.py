# Test that enumerating a dataset an object holds in a field types the element it yields.
import tensorflow as tf


def field_sink(x):
    return x * 2.0


def local_sink(x):
    return x * 2.0


class Loop:
    def __init__(self, dataset):
        self.loader = dataset

    def run(self):
        for step, element in enumerate(self.loader, 1):
            field_sink(element)


dataset = tf.data.Dataset.from_tensor_slices(tf.ones([4, 3])).batch(2)
Loop(dataset).run()

for step, element in enumerate(dataset, 1):
    local_sink(element)
    assert element.dtype == tf.float32 and element.shape == (2, 3)
