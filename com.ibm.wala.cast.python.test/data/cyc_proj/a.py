# One direction of a base-dependency cycle at module level: A's base comes from b (wala/ML#944).
from b import B


class A(B):
    pass
