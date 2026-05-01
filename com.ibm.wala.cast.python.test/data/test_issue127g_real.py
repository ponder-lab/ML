class C:
    def foo(self, val=0):
        return val


c = C()
a = c.foo(val=42)
