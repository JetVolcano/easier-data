from collections import deque
from collections.abc import Iterable, Sequence
from matplotlib.collections import PathCollection
from matplotlib.colors import Colormap
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.typing import ColorType, MarkerType
from numbers import Real
from pathlib import Path
from typing import TypeVar, Any
import matplotlib.pyplot as plt
import statistics as stats


T = TypeVar("T")

type ArrayLike[T] = list[T] | tuple[T, ...] | deque[T]


class Array1D:
    """
    1 Dimensional Array meant for 1 Dimensional Data
    """

    def __init__(self, data: ArrayLike[Real]) -> None:
        self.__data: ArrayLike[Real] = deque(data)
        self.__fig, self.__ax = plt.subplots()

    def data(self) -> ArrayLike[Real]:
        """
        Returns the data from the array

        :returntype ArrayLike[Real]:
        """
        return self.__data

    def append(self, obj: Real, /) -> None:
        """
        Appends an object to the array

        :param obj: The object to append
        :returntype None:
        """
        self.__data.append(obj)

    def appendleft(self, x: Real, /) -> None:
        """
        Appends an object to the left side of the array

        :param x: The object to append
        :returntype None:
        """
        self.__data.appendleft(x)

    def extend(self, iterable: Iterable[Real], /) -> None:
        """
        Extends an iterable to the array

        :param iterable: The iterable to extend
        :returntype None:
        """
        self.__data.extend(iterable)

    def extendleft(self, iterable: Iterable[Real], /) -> None:
        """
        Extends an iterable to the left side of the array

        :param iterable: The iterable to extend
        :returntype None:
        """
        self.__data.extendleft(iterable)

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
        return self.__ax.bar(self.__data, range(len(self.__data)))

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
        :param transparent: Wether or not the figure will be transparent
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
        figure: Figure = ...
        number: str | int = ...
        filename: str = f"figure_1.{suffix}"
        if dir is None:
            dir = Path(".\\figures\\")
        if isinstance(dir, str):
            dir = Path(dir)
        if not dir.exists():
            dir.mkdir()
        while Path(f"{dir.absolute().__str__()}\\{filename}").exists():
            figure, number, suffix = (
                filename.split("_")[0],
                filename.split("_")[1].split(".")[0],
                filename.split("_")[1].split(".")[1],
            )
            number = int(number) + 1
            filename = f"{figure}_{number}.{suffix}"
        path: Path = Path(dir.absolute().__str__() + "\\" + filename)
        self.__fig.savefig(path, transparent=transparent)

    def mean(self) -> Real:
        """
        Returns the mean of the data in the array

        :returntype Real:
        """
        return stats.mean(self.__data)

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
        return stats.median(self.__data)

    def quantiles(self) -> deque[Real]:
        """
        Returns the quantiles of the data in the array

        :returntype deque[Real]:
        """
        return deque(stats.quantiles(self.__data))

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

    def __init__(self, x: ArrayLike[Real], y: ArrayLike[Real]) -> None:
        self.__x: ArrayLike[Real] = deque(x)
        self.__y: ArrayLike[Real] = deque(y)
        self.__fig, self.__ax = plt.subplots()

    def x(self) -> ArrayLike[Real]:
        """
        Returns the data of x

        :returntype ArrayLike[Real]:
        """
        return self.__x

    def y(self) -> ArrayLike[Real]:
        """
        Returns the data of y

        :returntype ArrayLike[Real]
        """
        return self.__y

    def data(self) -> dict[ArrayLike[Real], ArrayLike[Real]]:
        """
        Returns a dict containing the x and y values

        :returntype dict[ArrayLike[Real], ArrayLike[Real]]
        """

        return {"x": self.__x, "y": self.__y}

    def append(self, x: Real, y: Real) -> None:
        """
        Append values to both the x and y posistions

        :param x: The value to append to the x posistion
        :param y: The value to append to the y posistion
        :returntype None:
        """
        self.__x.append(x)
        self.__y.append(y)

    def appendleft(self, x: Real, y: Real) -> None:
        """
        Append values to left of both x and y posistions

        :param x: The value to append to the left of the x posistion
        :param y: The value to append to the left of the y posistion
        :returntype None:
        """

        self.__x.appendleft(x)
        self.__y.appendleft(y)

    def extend(self, x: Iterable[Real], y: Iterable[Real]) -> None:
        """
        Extend an iterable both the x and y posistions

        :param x: The iterable to extend to the x posistion
        :param y: The iterable to extend to the y posistion
        :returntype None:
        """
        self.__x.extend(x)
        self.__y.extend(y)

    def extendleft(self, x: Iterable[Real], y: Iterable[Real]) -> None:
        """
        Extend an iterable to left of both x and y posistions

        :param x: The iterable to extend to the left of the x posistion
        :param y: The iterable to extend to the left of the y posistion
        :returntype None:
        """
        self.__x.extendleft(x)
        self.__y.extendleft(y)

    def plot(self) -> list[Line2D]:
        """
        Plot the data of the array

        :returntype list[Line2D]:
        """
        return self.__ax.plot(self.__x, self.__y)

    def scatter(
        self,
        s: ArrayLike[Real] | float | None = ...,
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
        :param transparent: Wether or not the figure will be transparent
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
        figure: Figure = ...
        number: str | int = ...
        filename: str = f"figure_1.{suffix}"
        if dir is None:
            dir = Path(".\\figures\\")
        if isinstance(dir, str):
            dir = Path(dir)
        if not dir.exists():
            dir.mkdir()
        while Path(f"{dir.absolute().__str__()}\\{filename}").exists():
            figure, number, suffix = (
                filename.split("_")[0],
                filename.split("_")[1].split(".")[0],
                filename.split("_")[1].split(".")[1],
            )
            number = int(number) + 1
            filename = f"{figure}_{number}.{suffix}"
        path: Path = Path(dir.absolute().__str__() + "\\" + filename)
        self.__fig.savefig(path, transparent=transparent)


class Array3D:
    """
    3 Dimensional Array meant for 3 Dimensional Data
    """

    def __init__(
        self, x: ArrayLike[Real], y: ArrayLike[Real], z: ArrayLike[Real]
    ) -> None:
        self.__x: ArrayLike[Real] = deque(x)
        self.__y: ArrayLike[Real] = deque(y)
        self.__z: ArrayLike[Real] = deque(z)
        self.__fig, self.__ax = plt.subplots(subplot_kw={"projection": "3d"})

    def x(self) -> ArrayLike[Real]:
        """
        Returns the data of x

        :returntype ArrayLike[Real]:
        """

        return self.__x

    def y(self) -> ArrayLike[Real]:
        """
        Returns the data of y

        :returntype ArrayLike[Real]:
        """
        return self.__y

    def z(self) -> ArrayLike[Real]:
        """
        Returns the data of z

        :returntype ArrayLike[Real]:
        """
        return self.__z

    def data(self) -> dict[ArrayLike[Real], ArrayLike[Real], ArrayLike[Real]]:
        """
        Returns a dict containing the x, y, and z values

        :returntype dict[ArrayLike[Real], ArrayLike[Real]]
        """

        return {"x": self.__x, "y": self.__y, "z": self.__z}

    def append(self, x: Real, y: Real, z: Real) -> None:
        """
        Append values to the x, y, and z posistions

        :param x: The value to append to the x posistion
        :param y: The value to append to the y posistion
        :param z: The value to append to the z posistion
        :returntype None:
        """
        self.__x.append(x)
        self.__y.append(y)
        self.__z.append(z)

    def appendleft(self, x: Real, y: Real, z: Real) -> None:
        """
        Append values to the left of the x, y, and z posistions

        :param x: The value to append to the left of the x posistion
        :param y: The value to append to the left of the y posistion
        :param z: The value to append to the left of the z posistion
        """
        self.__x.appendleft(x)
        self.__y.appendleft(y)
        self.__z.appendleft(z)

    def extend(self, x: Iterable[Real], y: Iterable[Real], z: Iterable[Real]) -> None:
        """
        Extend an iterable the x, y, and z posistions

        :param x: The iterable to extend to the x posistion
        :param y: The iterable to extend to the y posistion
        :param z: The iterable to extend to the z posistion
        :returntype None:
        """
        self.__x.extend(x)
        self.__y.extend(y)
        self.__z.extend(z)

    def extendleft(
        self, x: Iterable[Real], y: Iterable[Real], z: Iterable[Real]
    ) -> None:
        """
        Extend an iterable the left of the x, y, and z posistions

        :param x: The iterable to extend to the left of the x posistion
        :param y: The iterable to extend to the left of the y posistion
        :param z: The iterable to extend to the left of the z posistion
        :returntype None:
        """
        self.__x.extendleft(x)
        self.__y.extendleft(y)
        self.__z.extendleft(z)

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
        :param transparent: Wether or not the figure will be transparent
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
        figure: Figure = ...
        number: str | int = ...
        filename: str = f"figure_1.{suffix}"
        if dir is None:
            dir = Path(".\\figures\\")
        if isinstance(dir, str):
            dir = Path(dir)
        if not dir.exists():
            dir.mkdir()
        while Path(f"{dir.absolute().__str__()}\\{filename}").exists():
            figure, number, suffix = (
                filename.split("_")[0],
                filename.split("_")[1].split(".")[0],
                filename.split("_")[1].split(".")[1],
            )
            number = int(number) + 1
            filename = f"{figure}_{number}.{suffix}"
        path: Path = Path(dir.absolute().__str__() + "\\" + filename)
        self.__fig.savefig(path, transparent=transparent)
