from collections.abc import Sequence
from typing import Final, cast

from ._internal._types import _INSTANCE_CHECK_REAL, Real


class Point3D: ...


class Point2D:
    def __init__(self, x: Real, y: Real) -> None:
        """
        Parameters
        ----------
        x : Real
            The x value for the point
        y : Real
            The y value for the point
        """
        self.__BASE_ERRORS: Final[Sequence[TypeError]] = [
            TypeError("x must be a real number"),
            TypeError("y must be a real number"),
        ]
        self.__ERRORS: Final[Sequence[TypeError | None]] = [
            self.__BASE_ERRORS[0] if not isinstance(x, _INSTANCE_CHECK_REAL) else None,
            self.__BASE_ERRORS[1] if not isinstance(y, _INSTANCE_CHECK_REAL) else None,
        ]
        exceptions: list[TypeError] = []
        for i in range(len(self.__ERRORS)):
            if isinstance(self.__ERRORS[i], TypeError):
                exceptions.append(cast(TypeError, self.__ERRORS[i]))
        if len(exceptions) > 0:
            raise ExceptionGroup(f"{len(exceptions)} TypeError(s) occurred", exceptions)
        self.__x: Real = x
        self.__y: Real = y

    def __repr__(self) -> str:
        return f"Point2D(x={self.__x}, y={self.__y})"

    def __str__(self) -> str:
        return f"Point2D({self.__x}, {self.__y})"

    def __hash__(self) -> int:
        return hash((self.__x, self.__y))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point2D):
            return False
        return self.__dict__ == other.__dict__

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, Point2D):
            return False
        return self.__dict__ != other.__dict__

    @property
    def x(self) -> Real:
        return self.__x

    @property
    def y(self) -> Real:
        return self.__y

    @x.setter
    def x(self, value: Real) -> None:
        if not isinstance(value, _INSTANCE_CHECK_REAL):
            raise self.__BASE_ERRORS[0]
        self.__x = value

    @y.setter
    def y(self, value: Real) -> None:
        if not isinstance(value, _INSTANCE_CHECK_REAL):
            raise self.__BASE_ERRORS[1]
        self.__y = value
