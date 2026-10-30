from custom.callback import *
from tensorflow.python import keras


class Net(keras.Model):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = keras.layers.Dense(3)

    def call(self, inputs, training=None):
        return self.fc(inputs)

    def __train_step(self, x, y):
        predictions = self.call(x, training=True)
        self.loss_value = self.loss(y, predictions)
        return predictions

    def train_on_batch(self, x, y=None):
        return self.__train_step(x, y)
