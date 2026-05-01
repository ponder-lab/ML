class C:
    def __init__(self, val1):
        self.f = val1

    # Implementing `call` but invoking it directly like Keras Models
    def call(self, val2=0):
        res = self.f
        return val2


c = C(42)
a = c.call(val2=100)
