class C:
    def __init__(self, val1):
        self.f = val1

    def foo(self, val2):
        res = self.f
        return val2


c1 = C(42)
c2 = C(84)

a1 = c1.foo(100)
a2 = c2.foo(200)
