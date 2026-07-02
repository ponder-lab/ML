# Test for wala/ML#662 and wala/ML#106: a user subclass of `Model` imported via `from
# tensorflow.keras.models import Model` resolves its base class (the canonical instance type) in
# the class hierarchy instead of falling back to `object`.
from tensorflow.keras.models import Model


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()

    def call(self, inputs):
        return inputs


model = MyModel()
assert isinstance(model, Model)
