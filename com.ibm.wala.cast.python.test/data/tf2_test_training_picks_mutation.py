# Test https://github.com/wala/ML/issues/1249: a `training` value the analysis learns only once it is
# solved, the learning phase the program sets, picks the method a layer calls, and that method
# extends a list a slice was taken of. The analysis solves again and reaches the method, so the
# list's mutations it records are those of the call graph it finally builds.
import tensorflow as tf

SEEN = []


def consume(items):
    pass


class Picker(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()
        self.items = [1.0, 2.0]

    def mutating(self, x):
        self.items.extend([3.0])
        return x

    def plain(self, x):
        return x

    def call(self, x, training=None):
        head = self.items[:1]
        assert head == [1.0]
        consume(self.items)
        SEEN.append(training)
        return {1: self.mutating, None: self.plain}[training](x)


tf.keras.backend.set_learning_phase(1)
picker = Picker()
picker(tf.ones([2]))
assert SEEN == [1], SEEN
assert picker.items == [1.0, 2.0, 3.0], picker.items
