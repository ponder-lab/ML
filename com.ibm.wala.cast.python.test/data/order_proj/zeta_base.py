# The base module. It sorts LAST by path, so an ascending module order translates the subclasses'
# module before it (wala/ML#944).


class Base:
    def scale(self, x):
        return x * 2.0
