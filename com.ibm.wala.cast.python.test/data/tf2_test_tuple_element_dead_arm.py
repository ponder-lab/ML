# A constant-index read of a returned tuple keeps only the element's live arm (wala/ML#1153): the
# helper's `mode` is a constant at the call, so its other return is dead there, and reading the
# element through the tuple's points-to set would bring the dead return's rank-4 member in.
import tensorflow as tf


class Head:

    def project(self, x, mode="flat"):
        if mode == "flat":
            return tf.reshape(x, [1, 2, 3, 1])
        else:
            return tf.reshape(x, [2, 3, 1])

    def call(self, x):
        y = self.project(x, mode="cube")
        return y, x


def consume_element(t):
    return t


def consume_sliced(t):
    return t


head = Head()
logits, _ = head.call(tf.ones([6]))
consume_element(logits)
consume_sliced(logits[:, -1, :])
assert logits.shape == (2, 3, 1)
assert logits[:, -1, :].shape == (2, 1)
