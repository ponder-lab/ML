class C:
    def __init__(self, val1):
        self.f = val1

    def foo(self, val2):
        res1 = self.f
        return val2


c = C(42)
a = c.foo(100)
