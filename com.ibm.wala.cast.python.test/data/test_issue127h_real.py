class C:
    def __init__(self, val1):
        self.f = val1

    def __call__(self, val2=0):
        res = self.f
        return val2


c = C(42)
a = c(val2=100)
