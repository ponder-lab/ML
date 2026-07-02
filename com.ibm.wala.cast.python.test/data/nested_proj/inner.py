# Cross-module subclass for wala/ML#570/#571.
from messagepassing import MessagePassing


class Inner(MessagePassing):
    def __init__(self):
        super(Inner, self).__init__()

    def call(self, inputs):
        return self.propagate(inputs)
