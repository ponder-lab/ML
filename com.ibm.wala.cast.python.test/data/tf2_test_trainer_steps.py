# Fixture for wala/ML#1051: a training loop reached through a generator of steps.
#
# `Trainer.__call__` iterates the generator `_steps`, which wraps a nested closure in `tf.function`
# and calls it; the closure calls the instance's `_forward`, which hands the gradients to an
# accumulator. Each sink marks one link of the chain.
import tensorflow as tf


def consume_call(x):
    pass


def consume_steps(x):
    pass


def consume_closure(x):
    pass


def consume_forward(x):
    pass


class Accumulator:
    def __call__(self, gradients):
        consume_forward(gradients[0])


class Trainer:
    def __init__(self):
        self._accumulator = Accumulator()

    def __call__(self, dataset):
        consume_call(tf.constant(1.0))
        for i, loss in enumerate(self._steps(dataset)):
            pass

    def _steps(self, dataset):
        consume_steps(tf.constant(2.0))
        iterator = iter(dataset)

        def _forward():
            batch = next(iterator)
            consume_closure(batch)
            return self._forward(batch)

        forward_fn = tf.function(_forward)
        for _ in range(2):
            yield forward_fn()

    def _forward(self, batch):
        gradients = [batch * 2.0]
        self._accumulator(gradients)
        return batch


dataset = tf.data.Dataset.from_tensor_slices(tf.ones([4, 3]))
trainer = Trainer()
trainer(dataset)
