# The classes a parameterized test selects between (wala/ML#868): each has a
# method so receiver dispatch through the bound parameter is observable.
class A:
    def tag(self):
        return 1


class B:
    def tag(self):
        return 2
