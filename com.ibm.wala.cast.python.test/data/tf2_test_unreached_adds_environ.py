# A function the call graph does not reach may still run. Its `update` on an attribute of a module
# other than TensorFlow's backend may add an element to any dict.
import os


def consume(c):
    pass


def override(settings):
    os.environ.update(settings)


c = {"a": 1, "b": 2}
assert len(c) == 2 and c["a"] == 1
consume(c)
