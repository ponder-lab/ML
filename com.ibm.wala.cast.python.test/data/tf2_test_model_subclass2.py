# Test for wala/ML#662: a user subclass of `Model` imported via `from tensorflow.keras.models
# import Model` currently still falls back to `object` in the class hierarchy: the canonical
# `tensorflow/keras/models/Model` instance type carries the `__call__`/`call` summary bodies, and a
# method-less shell under that name would shadow them (see the CAUTION in `tensorflow.xml`).
# TODO: The base should resolve to the canonical type once shells carry the summary's own methods
# per wala/ML#106 (https://github.com/wala/ML/issues/106).
from tensorflow.keras.models import Model


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()

    def call(self, inputs):
        return inputs


model = MyModel()
assert isinstance(model, Model)
