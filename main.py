import matplotlib.axes
import matplotlib.container
import matplotlib.pyplot as plot
import matplotlib.lines
import numpy
import seaborn
import statistics as stats
from collections import deque
from collections.abc import Iterable
from numbers import Number
from typing import Any


class DataSet:
    def __init__(self, data: list[Number] | deque[Number]) -> None:
        self.__data: deque[Number] = deque(data)

    @property
    def mean(self) -> float:
        return stats.mean(self.__data)
    
    @property
    def median(self) -> float:
        return stats.median(self.__data)
    
    @property
    def quantiles(self) -> list[float]:
        return stats.quantiles(self.__data)
    
    @property
    def q1(self) -> float:
        return self.quantiles[0]
    
    @property
    def q3(self) -> float:
        return self.quantiles[2]
    
    @property
    def iqr(self) -> float:
        return self.q3-self.q1
    
    @property
    def stdev(self) -> float:
        return stats.stdev(self.__data)

class Array1D:
    def __init__(self, x: list[Any] | deque[Any]) -> None:
        self.x: deque[Any] = deque(x)
        self.fig, self.ax = plot.subplots()
        self.set_xlabel = self.ax.set_xlabel
        self.set_ylabel = self.ax.set_ylabel
        self.set_xlabel("x")
        self.set_ylabel("y")

    def __repr__(self) -> str:
        return f"Array2D(x={self.x})"

    def plot(self) -> list[matplotlib.lines.Line2D]:
        return self.ax.plot(self.x)

    def boxplot(self) -> dict[str, Any]:
        return self.ax.boxplot(self.x)


class Array2D:
    def __init__(self, x: list[Any] | deque[Any], y: list[Any] | deque[Any]) -> None:
        self.x: deque[Any] = deque(x)
        self.y: deque[Any] = deque(y)
        self.fig, self.ax = plot.subplots()
        self.set_xlabel = self.ax.set_xlabel
        self.set_ylabel = self.ax.set_ylabel
        self.set_xlabel("x")
        self.set_ylabel("y")

    def __repr__(self) -> str:
        return f"Array2D(x={self.x}, y={self.y})"

    def plot(self) -> list[matplotlib.lines.Line2D]:
        return self.ax.plot(self.x, self.y)

    def bar(self) -> matplotlib.container.BarContainer:
        return self.ax.bar(self.x, self.y)
