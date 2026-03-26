class C:
    def __init__(self, val):
        self.f = val

    def foo(self):
        return self.f


c = C(42)
a = c.foo()
