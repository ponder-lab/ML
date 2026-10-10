class Layer:
    def __init__(self):
        self.scale = 2.0

    def __call__(self, x):
        return x * self.scale
