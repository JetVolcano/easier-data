from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np
from filelock import FileLock
from matplotlib.collections import PathCollection
from matplotlib.colors import Colormap
from matplotlib.container import BarContainer
from matplotlib.lines import Line2D
from matplotlib.typing import ColorType, MarkerType
from scipy import stats

from ._types import ArrayLike, check_type


if TYPE_CHECKING:
    from scipy.stats._stats_py import ModeResult


class Array1D:
    """
    1 Dimensional Array meant for 1 Dimensional Data
    """

    __hash__: ClassVar[None] = None

    def __init__(self, data: ArrayLike[Real]) -> None:
        if len(data) == 0:
            raise ValueError("Data cannot be empty.")
        if not check_type(data, Real):
            raise TypeError("Data must only contain real numbers.")
        self.__data: np.ndarray = np.array(data)
        self._type: type = type(data)
        self.__fig, self.__ax = plt.subplots()

    def __repr__(self) -> str:
        """
        Returns the representation of the 1 Dimensional Array

        :returntype str:
        """
        return f"Array1D(data={self._type(self.__data) if self._type != np.ndarray else self.__data!r})"

    def __str__(self) -> str:
        """
        Returns the string representation of the 1 Dimensional Array

        :returntype str:
        """
        return f"Array1D({self._type(self.__data) if self._type != np.ndarray else self.__data!r})"

    def __eq__(self, other: Array1D) -> Any:
        return self.__data == other.__data

    def __ne__(self, other: Array1D) -> Any:
        return self.__data != other.__data

    def __gt__(self, other: Array1D) -> Any:
        return self.__data > other.__data

    def __ge__(self, other: Array1D) -> Any:
        return self.__data >= other.__data

    def __lt__(self, other: Array1D) -> Any:
        return self.__data < other.__data

    def __le__(self, other: Array1D) -> Any:
        return self.__data <= other.__data

    @property
    def data(self) -> np.ndarray:
        """
        Returns the data from the array

        :returntype np.ndarray:
        """
        return self.__data

    def append(self, obj: Real, /) -> None:
        """
        Appends an object to the array

        :param obj: The object to append
        :returntype None:
        """
        self.__data = np.append(self.__data, obj)

    def appendleft(self, x: Real, /) -> None:
        """
        Appends an object to the left side of the array

        :param x: The object to append
        :returntype None:
        """
        self.__data = np.insert(self.__data, 0, x)

    def extend(self, iterable: Iterable[Real], /) -> None:
        """
        Extends an iterable to the array

        :param iterable: The iterable to extend
        :returntype None:
        """
        self.__data = np.concatenate((self.__data, iterable))

    def extendleft(self, iterable: Iterable[Real], /) -> None:
        """
        Extends an iterable to the left side of the array

        :param iterable: The iterable to extend
        :returntype None:
        """
        self.__data = np.concatenate((np.array(iterable)[::-1], self.__data))

    def plot(self) -> list[Line2D]:
        """
        Plots the data of the array

        :returntype list[Line2D]:
        """
        return self.__ax.plot(self.__data)

    def bar(self) -> BarContainer:
        """
        Make a bar plot of the array

        :returntype BarContainer:
        """
        return self.__ax.bar(range(len(self.__data)), self.__data)

    def boxplot(self) -> dict[str, Any]:
        """
        Make a boxplot of the array

        :returntype dict[str, Any]:
        """
        return self.__ax.boxplot(self.__data)

    def show(self) -> None:
        """
        Shows the current figure

        :returntype None:
        """
        self.__fig.show()

    def save(
        self,
        dir: str | Path | None = None,
        suffix: str = "svg",
        *,
        transparent: bool | None = None,
    ) -> None:
        """
        :param dir: A directory to the path that the figure will be saved
        :param suffix: The suffix for the saved figure
        :param transparent: Whether or not the figure will be transparent
        :returntype None:
        """
        formats: deque[str] = deque(
            (
                "eps",
                "jpeg",
                "jpg",
                "pdf",
                "pgf",
                "png",
                "ps",
                "raw",
                "rgba",
                "svg",
                "svgz",
                "tif",
                "tiff",
                "webp",
            )
        )
        if suffix not in formats:
            raise ValueError(
                f"Format '{suffix}' is not supported (supported formats: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff, webp)"
            )
        number = 1
        filename: str = f"figure_1.{suffix}"
        if dir is None:
            dir = Path(".\\figures\\")
        if isinstance(dir, str):
            dir = Path(dir)
        if not dir.exists():
            dir.mkdir()

        lock_path: Path = dir / "save.lock"
        with FileLock(str(lock_path)):
            while (dir / f"figure_{number}.{suffix}").exists():
                number += 1
            filename = f"figure_{number}.{suffix}"
            path: Path = dir / filename
            self.__fig.savefig(path, transparent=transparent)

    def mean(self) -> Real:
        """
        Returns the mean of the data in the array

        :returntype Real:
        """
        return self.__data.mean()

    def avg(self) -> Real:
        """
        Returns the average of the data in the array

        :returntype Real:
        """
        return self.mean()

    def median(self) -> Real:
        """
        Returns the median of the data in the array

        :returntype Real:
        """
        return np.median(self.__data)

    def mode(self) -> "ModeResult":
        """
        Returns the mode of the data in the array

        :returntype ModeResult:
        """
        return stats.mode(self.__data)

    def std(self) -> Real:
        """
        Returns the standard deviation of the data in the array

        :returntype Real:
        """
        return self.data.std()

    def stddev(self) -> Real:
        """
        Returns the standard deviation of the data in the array

        :returntype Real:
        """
        return self.data.std()

    def quantiles(self) -> np.floating[Any]:
        """
        Returns the quantiles of the data in the array

        :returntype floating[Any]:
        """
        return np.quantile(self.__data, [0.25, 0.5, 0.75])

    def q1(self) -> Real:
        """
        Returns the first quartile of the data in the array

        :returntype Real:
        """
        return self.quantiles()[0]

    def q3(self) -> Real:
        """
        Returns the third quartile of the data in the array

        :returntype Real:
        """
        return self.quantiles()[2]

    def iqr(self) -> Real:
        """
        Returns the interquartile range of the data in the array

        :returntype Real:
        """
        return self.q3() - self.q1()


class Array2D:
    """
    2 Dimensional Array meant for 2 Dimensional Data
    """

    __hash__: ClassVar[None] = None

    def __init__(self, x: ArrayLike[Real], y: ArrayLike[Real]) -> None:
        if len(x) == 0 or len(y) == 0:
            raise ValueError("Data cannot be empty.")
        if len(x) != len(y):
            raise ValueError("x and y must be the same length.")
        x_exception: TypeError = TypeError("x must contain only real numbers")
        y_exception: TypeError = TypeError("y must contain only real numbers")
        exceptions: dict[TypeError, bool] = {
            x_exception: not check_type(x, Real),
            y_exception: not check_type(y, Real),
        }
        if any(exceptions.values()):
            raise ExceptionGroup(
                f"{sum(exceptions.values())} TypeError(s) occurred",
                [exception for exception in exceptions if exceptions[exception]],
            )
        self.__x: np.ndarray = np.array(x)
        self._xtype: type = type(x)
        self.__y: np.ndarray = np.array(y)
        self._ytype: type = type(y)
        self.__fig, self.__ax = plt.subplots()

    def __repr__(self) -> str:
        """
        Returns the representation of the 2 Dimensional Array

        :returntype str:
        """
        return f"Array2D(x={self._xtype(self.__x) if self._xtype != np.ndarray else self.__x!r}, y={self._ytype(self.__y) if self._ytype != np.ndarray else self.__y!r})"

    def __str__(self) -> str:
        """
        Returns the string representation of the 2 Dimensional Array

        :returntype str:
        """
        return f"Array2D({self._xtype(self.__x) if self._xtype != np.ndarray else self.__x!r}, {self._ytype(self.__y) if self._ytype != np.ndarray else self.__y!r})"

    def __eq__(self, other: Array2D) -> Any:
        return np.array([self.__x, self.__y]) == np.array([other.__x, other.__y])

    def __ne__(self, other: Array2D) -> Any:
        return np.array([self.__x, self.__y]) != np.array([other.__x, other.__y])

    def __gt__(self, other: Array2D) -> Any:
        return np.array([self.__x, self.__y]) > np.array([other.__x, other.__y])

    def __ge__(self, other: Array2D) -> Any:
        return np.array([self.__x, self.__y]) >= np.array([other.__x, other.__y])

    def __lt__(self, other: Array2D) -> Any:
        return np.array([self.__x, self.__y]) < np.array([other.__x, other.__y])

    def __le__(self, other: Array2D) -> Any:
        return np.array([self.__x, self.__y]) <= np.array([other.__x, other.__y])

    @property
    def x(self) -> np.ndarray:
        """
        Returns the data of x

        :returntype np.ndarray:
        """
        return self.__x

    @property
    def y(self) -> np.ndarray:
        """
        Returns the data of y

        :returntype np.ndarray:
        """
        return self.__y

    @property
    def data(self) -> dict[str, np.ndarray]:
        """
        Returns a dict containing the x and y values

        :returntype dict[str, np.ndarray]
        """

        return {"x": self.__x, "y": self.__y}

    def append(self, x: Real, y: Real) -> None:
        """
        Append values to both the x and y posistions

        :param x: The value to append to the x posistion
        :param y: The value to append to the y posistion

        :returntype None:
        """
        self.__x = np.append(self.__x, x)
        self.__y = np.append(self.__y, y)

    def appendleft(self, x: Real, y: Real) -> None:
        """
        Append values to left of both x and y positions

        :param x: The value to append to the left of the x position
        :param y: The value to append to the left of the y position

        :returntype None:
        """

        self.__x = np.insert(self.__x, 0, x)
        self.__y = np.insert(self.__y, 0, y)

    def extend(self, x: Iterable[Real], y: Iterable[Real]) -> None:
        """
        Extend an iterable both the x and y positions

        :param x: The iterable to extend to the x position
        :param y: The iterable to extend to the y position
        :returntype None:
        """
        self.__x = np.concatenate((self.__x, x))
        self.__y = np.concatenate((self.__y, y))

    def extendleft(self, x: Iterable[Real], y: Iterable[Real]) -> None:
        """
        Extend an iterable to left of both x and y positions

        :param x: The iterable to extend to the left of the x position
        :param y: The iterable to extend to the left of the y position
        :returntype None:
        """
        self.__x = np.concatenate((np.array(x)[::-1], self.__x))
        self.__y = np.concatenate((np.array(y)[::-1], self.__y))

    def plot(self) -> list[Line2D]:
        """
        Plot the data of the array

        :returntype list[Line2D]:
        """
        return self.__ax.plot(self.__x, self.__y)

    def scatter(
        self,
        s: ArrayLike[Real] | float | None = None,
        c: ArrayLike | ColorType | Sequence[ColorType] | None = None,
        marker: MarkerType | None = None,
        cmap: str | Colormap | None = None,
        alpha: float | None = None,
    ) -> PathCollection:
        """
        Returns a scatter plot of the array

        :param s: Size of the marker
        :param marker: The marker of the plot
        :param cmap: The colormap to plot
        :param alpha: The alpha of all the markers

        :returntype PathCollection:
        """
        return self.__ax.scatter(
            self.__x, self.__y, s, c, marker=marker, alpha=alpha, cmap=cmap
        )

    def bar(self) -> BarContainer:
        """
        Make a bar plot of the array

        :returntype BarContainer:
        """
        return self.__ax.bar(self.__x, self.__y)

    def show(self) -> None:
        """
        Shows the current figure

        :returntype None:
        """
        self.__fig.show()

    def save(
        self,
        dir: str | Path | None = None,
        suffix: str = "svg",
        *,
        transparent: bool | None = None,
    ) -> None:
        """
        :param dir: A directory to the path that the figure will be saved
        :param suffix: The suffix for the saved figure
        :param transparent: Whether or not the figure will be transparent
        :returntype None:
        """
        formats: deque[str] = deque(
            (
                "eps",
                "jpeg",
                "jpg",
                "pdf",
                "pgf",
                "png",
                "ps",
                "raw",
                "rgba",
                "svg",
                "svgz",
                "tif",
                "tiff",
                "webp",
            )
        )
        if suffix not in formats:
            raise ValueError(
                f"Format '{suffix}' is not supported (supported formats: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff, webp)"
            )
        number = 1
        filename: str = f"figure_1.{suffix}"
        if dir is None:
            dir = Path(".\\figures\\")
        if isinstance(dir, str):
            dir = Path(dir)
        if not dir.exists():
            dir.mkdir()

        lock_path: Path = dir / "save.lock"
        with FileLock(str(lock_path)):
            while (dir / f"figure_{number}.{suffix}").exists():
                number += 1
            filename = f"figure_{number}.{suffix}"
            path: Path = dir / filename
            self.__fig.savefig(path, transparent=transparent)


class Array3D:
    """
    3 Dimensional Array meant for 3 Dimensional Data
    """

    __hash__: ClassVar[None] = None

    def __init__(
        self, x: ArrayLike[Real], y: ArrayLike[Real], z: ArrayLike[Real]
    ) -> None:
        if len(x) == 0 or len(y) == 0 or len(z) == 0:
            raise ValueError("Data cannot be empty.")
        if len(x) != len(y) or len(y) != len(z):
            raise ValueError("x, y, and z must be the same length.")
        x_exception: TypeError = TypeError("x must contain only real numbers")
        y_exception: TypeError = TypeError("y must contain only real numbers")
        z_exception: TypeError = TypeError("z must contain only real numbers")
        exceptions: dict[TypeError, bool] = {
            x_exception: not check_type(x, Real),
            y_exception: not check_type(y, Real),
            z_exception: not check_type(z, Real),
        }
        if any(exceptions.values()):
            raise ExceptionGroup(
                f"{sum(exceptions.values())} TypeError(s) occurred",
                [exception for exception in exceptions if exceptions[exception]],
            )
        self.__x: np.ndarray = np.array(x)
        self._xtype: type = type(x)
        self.__y: np.ndarray = np.array(y)
        self._ytype: type = type(y)
        self.__z: np.ndarray = np.array(z)
        self._ztype: type = type(z)
        self.__fig, self.__ax = plt.subplots(subplot_kw={"projection": "3d"})

    def __repr__(self) -> str:
        """
        Returns the representation of the 3 Dimensional Array

        :returntype str:
        """
        return f"Array3D(x={self._xtype(self.__x) if self._xtype != np.ndarray else self.__x!r}, y={self._ytype(self.__y) if self._ytype != np.ndarray else self.__y!r}, z={self._ztype(self.__z) if self._ztype != np.ndarray else self.__z!r})"

    def __str__(self) -> str:
        """
        Returns the string representation of the 3 Dimensional Array

        :returntype str:
        """
        return f"Array3D({self._xtype(self.__x) if self._xtype != np.ndarray else self.__x!r}, {self._ytype(self.__y) if self._ytype != np.ndarray else self.__y!r}, {self._ztype(self.__z) if self._ztype != np.ndarray else self.__z!r})"

    @property
    def x(self) -> np.ndarray:
        """
        Returns the data of x

        :returntype np.ndarray:
        """
        return self.__x

    @property
    def y(self) -> np.ndarray:
        """
        Returns the data of y

        :returntype np.ndarray:
        """
        return self.__y

    @property
    def z(self) -> np.ndarray:
        """
        Returns the data of z

        :returntype np.ndarray:
        """
        return self.__z

    @property
    def data(self) -> dict[str, np.ndarray]:
        """
        Returns a dict containing the x, y, and z values

        :returntype dict[str, np.ndarray]
        """
        return {"x": self.__x, "y": self.__y, "z": self.__z}

    def append(self, x: Real, y: Real, z: Real) -> None:
        """
        Append values to the x, y, and z posisions

        :param x: The value to append to the x position
        :param y: The value to append to the y position
        :param z: The value to append to the z position
        :returntype None:
        """
        self.__x = np.append(self.__x, x)
        self.__y = np.append(self.__y, y)
        self.__z = np.append(self.__z, z)

    def appendleft(self, x: Real, y: Real, z: Real) -> None:
        """
        Append values to the left of the x, y, and z positions

        :param x: The value to append to the left of the x position
        :param y: The value to append to the left of the y position
        :param z: The value to append to the left of the z position
        """
        self.__x = np.insert(self.__x, 0, x)
        self.__y = np.insert(self.__y, 0, y)
        self.__z = np.insert(self.__z, 0, z)

    def extend(self, x: Iterable[Real], y: Iterable[Real], z: Iterable[Real]) -> None:
        """
        Extend an iterable the x, y, and z positions

        :param x: The iterable to extend to the x position
        :param y: The iterable to extend to the y position
        :param z: The iterable to extend to the z position
        :returntype None:
        """
        self.__x = np.concatenate((self.__x, x))
        self.__y = np.concatenate((self.__y, y))
        self.__z = np.concatenate((self.__z, z))

    def extendleft(
        self, x: Iterable[Real], y: Iterable[Real], z: Iterable[Real]
    ) -> None:
        """
        Extend an iterable the left of the x, y, and z positions

        :param x: The iterable to extend to the left of the x position
        :param y: The iterable to extend to the left of the y position
        :param z: The iterable to extend to the left of the z position
        :returntype None:
        """
        self.__x = np.concatenate((np.array(x)[::-1], self.__x))
        self.__y = np.concatenate((np.array(y)[::-1], self.__y))
        self.__z = np.concatenate((np.array(z)[::-1], self.__z))

    def save(
        self,
        dir: str | Path | None = None,
        suffix: str = "svg",
        *,
        transparent: bool | None = None,
    ) -> None:
        """
        :param dir: A directory to the path that the figure will be saved
        :param suffix: The suffix for the saved figure
        :param transparent: Whether or not the figure will be transparent
        :returntype None:
        """
        formats: deque[str] = deque(
            (
                "eps",
                "jpeg",
                "jpg",
                "pdf",
                "pgf",
                "png",
                "ps",
                "raw",
                "rgba",
                "svg",
                "svgz",
                "tif",
                "tiff",
                "webp",
            )
        )
        if suffix not in formats:
            raise ValueError(
                f"Format '{suffix}' is not supported (supported formats: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff, webp)"
            )
        number = 1
        filename: str = f"figure_1.{suffix}"
        if dir is None:
            dir = Path(".\\figures\\")
        if isinstance(dir, str):
            dir = Path(dir)
        if not dir.exists():
            dir.mkdir()

        lock_path: Path = dir / "save.lock"
        with FileLock(str(lock_path)):
            while (dir / f"figure_{number}.{suffix}").exists():
                number += 1
            filename = f"figure_{number}.{suffix}"
            path: Path = dir / filename
            self.__fig.savefig(path, transparent=transparent)
