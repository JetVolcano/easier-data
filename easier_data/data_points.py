import statistics as stats
import matplotlib.axes
import matplotlib.container
import matplotlib.pyplot as plot
import matplotlib.lines
from collections import deque
from collections.abc import Iterable
from numbers import Number
from typing import Any


class DataSet:
    def __init__(self, data: list[Number] | deque[Number]) -> None:
        self.__data: deque[Number] = deque(data)
        self.fig, self.ax = plot.subplots()
        self.set_xlabel = self.ax.set_xlabel
        self.set_ylabel = self.ax.set_ylabel
        self.set_xlabel("x")
        self.set_ylabel("y")

    def __repr__(self) -> str:
        return f"DataSet(data={self.__data})"

    def __str__(self) -> str:
        return f"DataSet({self.__data})"

    def append(self, x: Number) -> None:
        self.__data.append(x)

    def appendleft(self, x: Number) -> None:
        self.__data.appendleft(x)

    def extend(self, iterable: Iterable[Number]) -> None:
        self.__data.extend(iterable)

    def extendleft(self, iterable: Iterable[Number]) -> None:
        self.__data.extendleft(iterable)

    def mean(self) -> float:
        return stats.mean(self.__data)

    def median(self) -> float:
        return stats.median(self.__data)

    def quantiles(self) -> list[float]:
        return stats.quantiles(self.__data)

    def q1(self) -> float:
        return self.quantiles[0]

    def q3(self) -> float:
        return self.quantiles[2]

    def iqr(self) -> float:
        return self.q3 - self.q1

    def stdev(self) -> float:
        return stats.stdev(self.__data)

    def mad(self) -> float:
        return stats.mean([abs(val - self.mean) for val in self.__data])

    def plot(self) -> matplotlib.lines.Line2D:
        return self.ax.plot(self.__data)

    def boxplot(self) -> dict[str, Any]:
        return self.ax.boxplot(self.__data)
