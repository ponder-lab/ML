import tensorflow as tf
from tensorflow.keras import Model


# The wala/ML#657 shape: a `tf.keras.Model` subclass that subclasses the *bare* imported name
# `Model` (the ubiquitous `from tensorflow.keras import Model; class X(Model)` idiom) and is reached
# via `model(x)` callable dispatch. Analyzed together with a second module that defines its own
# `class Model` (`tf2_657_collide.py`), the bare base name previously mis-resolved across modules,
# nulling `MyModel`'s superclass and dropping `MyModel.call` from the call graph.
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d = tf.keras.layers.Dense(10)

    def call(self, x):
        return self.d(x)


model = MyModel()


@tf.function
def train_step(images):
    return model(images)


train_step(tf.ones([1, 10]))
