from .points import Point2D


class Line:
    def __init__(self, p1: Point2D, p2: Point2D) -> None:
        self._point1: Point2D = p1
        self._point2: Point2D = p2
