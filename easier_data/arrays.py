import matplotlib.pyplot as plt
import numpy
import statistics as stats
from collections import deque
from collections.abc import Iterable
from matplotlib.container import BarContainer
from matplotlib.lines import Line2D
from numbers import Real
from typing import TypeVar, Any


T = TypeVar("T")

type ArrayLike[T] = list[T] | tuple[T, ...] | deque[T]


class Array1D:
    def __init__(self, data: ArrayLike[Real]) -> None:
        self.__data: ArrayLike[Real] = deque(data)
        self.__fig, self.__ax = plt.subplots()

    def data(self) -> ArrayLike[Real]:
        return self.__data

    def append(self, obj: Real, /) -> None:
        self.__data.append(obj)

    def appendleft(self, x: Real, /) -> None:
        self.__data.appendleft(x)

    def extend(self, iterable: Iterable[Real], /) -> None:
        self.__data.extend(iterable)

    def extendleft(self, iterable: Iterable[Real], /) -> None:
        self.__data.extendleft(iterable)

    def plot(self) -> list[Line2D]:
        return self.__ax.plot(self.__data)

    def bar(self) -> BarContainer:
        return self.__ax.bar(self.__data, range(len(self.__data)))

    def boxplot(self) -> dict[str, Any]:
        return self.__ax.boxplot(self.__data)

    def mean(self) -> Real:
        return stats.mean(self.__data)

    def avg(self) -> Real:
        return self.mean()

    def median(self) -> Real:
        return stats.median(self.__data)

    def quantiles(self) -> deque[Real]:
        return deque(stats.quantiles(self.__data))

    def q1(self) -> Real:
        return self.quantiles()[0]

    def q3(self) -> Real:
        return self.quantiles()[2]

    def iqr(self) -> Real:
        return self.q3() - self.q1()


class Array2D:
    def __init__(self, x: ArrayLike[Real], y: ArrayLike[Real]) -> None:
        self.__x: ArrayLike[Real] = deque(x)
        self.__y: ArrayLike[Real] = deque(y)

    def x(self) -> ArrayLike[Real]:
        return self.__x

    def y(self) -> ArrayLike[Real]:
        return self.__y

    def data(self) -> dict[ArrayLike[Real], ArrayLike[Real]]:
        return {"x": self.__x, "y": self.__y}


class Array3D:
    def __init__(
        self, x: ArrayLike[Real], y: ArrayLike[Real], z: ArrayLike[Real]
    ) -> None:
        self.__x: ArrayLike[Real] = deque(x)
        self.__y: ArrayLike[Real] = deque(y)
        self.__z: ArrayLike[Real] = deque(z)

    def x(self) -> ArrayLike[Real]:
        return self.__x

    def y(self) -> ArrayLike[Real]:
        return self.__y

    def z(self) -> ArrayLike[Real]:
        return self.__z

    def data(self) -> dict[ArrayLike[Real], ArrayLike[Real], ArrayLike[Real]]:
        return {"x": self.__x, "y": self.__y, "z": self.__z}
