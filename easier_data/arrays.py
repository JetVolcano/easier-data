import matplotlib.axes
import matplotlib.container
import matplotlib.pyplot as plot
import matplotlib.lines
import numpy
from collections import deque
from typing import Any


class Array1D:
    def __init__(self, x: list[Any] | deque[Any]) -> None:
        self.x: deque[Any] = deque(x)
        self.fig, self.ax = plot.subplots()
        self.set_xlabel = self.ax.set_xlabel
        self.set_ylabel = self.ax.set_ylabel
        self.set_xlabel("x")
        self.set_ylabel("y")

    def __repr__(self) -> str:
        return f"Array1D(x={self.x})"

    def __str__(self) -> str:
        return f"Array1D({self.x})"

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

    def __str__(self) -> str:
        return f"Array2D({self.x}, {self.y})"

    def plot(self) -> list[matplotlib.lines.Line2D]:
        return self.ax.plot(self.x, self.y)

    def bar(self) -> matplotlib.container.BarContainer:
        return self.ax.bar(self.x, self.y)

    def boxplot(self) -> dict[str, Any]:
        return self.ax.boxplot([self.x, self.y])
