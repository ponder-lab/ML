class C:
    def __init__(self, f):
        self.f = f

    def __call__(self):
        return self.f


c1 = C(42)
c2 = C(84)
v1 = c1()
v2 = c2()
