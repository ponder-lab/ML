# A constant-index read of a tuple that a concatenation builds, rather than a tuple display. The
# analysis allocates such a tuple at the `+` operation, which is not a `new` instruction, so the
# read can't be resolved through the tuple's display and falls back to its points-to set.
import tensorflow as tf


class Head:

    def project(self, x, mode="flat"):
        if mode == "flat":
            return tf.reshape(x, [1, 2, 3, 1])
        else:
            return tf.reshape(x, [2, 3, 1])

    def call(self, x):
        y = self.project(x, mode="cube")
        return (y,) + (x,)


def consume_sliced(t):
    return t


head = Head()
logits, _ = head.call(tf.ones([6]))
consume_sliced(logits[:, -1, :])
assert logits.shape == (2, 3, 1)
assert logits[:, -1, :].shape == (2, 1)
