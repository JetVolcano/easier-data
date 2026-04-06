from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import matplotlib.pyplot as plt
import numpy as np
from filelock import FileLock
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.colors import Colormap
from matplotlib.container import BarContainer
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.typing import ColorType, MarkerType

# from mpl_toolkits.mplot3d.art3d import Line3D, Path3DCollection
from scipy import stats
from scipy.interpolate import CubicSpline

from ._internal._types import ArrayLike, Real
from ._internal.errors import NDArrayErrors


if TYPE_CHECKING:
    from scipy.stats._stats_py import ModeResult

_FIGURE_FORMATS: Final[tuple[str, ...]] = (
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


class Array3D:
    """
    3 Dimensional Array meant for 3 Dimensional Data
    """

    def __init__(
        self, x: ArrayLike[Real], y: ArrayLike[Real], z: ArrayLike[Real]
    ) -> None:
        """
        Parameters
        ----------
        x : ArrayLike[Real]
            1-Dimensional Data for the x-axis
        y : ArrayLike[Real]
            1-Dimensional Data for the y-axis
        z : ArrayLike[Real]
            1-Dimensional Data for the z-axis

        Raises
        ------
        ValueError
            If any of the arrays are empty.
        ValueError
            If the arrays are not the same length.
        """
        NDArrayErrors.ARR3D(x, y, z)
        self._points: np.ndarray = np.column_stack(cast(Sequence[ArrayLike], (x, y, z)))
        self._original_x: str = f"{x!r}"
        self._original_y: str = f"{y!r}"
        self._original_z: str = f"{z!r}"
        self._fig: Figure | None = None
        self._ax: Axes | None = None

    def __repr__(self) -> str:
        return (
            f"Array3D(x={self._original_x}, y={self._original_y}, z={self._original_z})"
        )

    def __str__(self) -> str:
        return f"Array3D({self._original_x}, {self._original_y}, {self._original_z})"

    @property
    def x(self) -> np.ndarray:
        """1-Dimensional Data for the x-axis"""
        return self._points[:, 0]

    @property
    def y(self) -> np.ndarray:
        """1-Dimensional Data for the y-axis"""
        return self._points[:, 1]

    @property
    def z(self) -> np.ndarray:
        """1-Dimensional Data for the z-axis"""
        return self._points[:, 2]

    @property
    def data(self) -> np.ndarray:
        return self._points

    def plot(self): ...
    def bar(self): ...
    def stem(self): ...
    def scatter(self): ...
    def quiver(self): ...
    def show(self) -> None:
        """
        Shows the current figure
        """
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots(subplot_kw={"projection": "3d"})

        self._fig = cast(Figure, self._fig)
        self._ax = cast(Axes, self._ax)
        self._fig.show()

    def save(
        self,
        dir: str | Path | None = None,
        suffix: str = "svg",
        *,
        transparent: bool | None = None,
    ) -> None:
        """Saves the current figure to a file.

        Parameters
        ----------
        dir : str | Path | None, optional, default=None
            The path that the figure will be saved.
        suffix : str, optional, default='svg'
            The suffix for the file.
        transparent : bool | None, optional, default=None
            Whether the image is transparent.

        Raises
        ------
        ValueError
            Suffix is not supported.
        """
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots(subplot_kw={"projection": "3d"})

        self._fig = cast(Figure, self._fig)
        self._ax = cast(Axes, self._ax)
        if suffix not in _FIGURE_FORMATS:
            raise ValueError(
                f"Format: '{suffix}' is not supported, (supported formats: {', '.join(_FIGURE_FORMATS)})"
            )
        number: int = 1
        filename: str = f"figure_{number}.{suffix}"
        if dir is None:
            dir = Path("figures")
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
            self._fig.savefig(path, transparent=transparent)


class Array2D:
    """
    Array2D
    -------

    Takes in 2 -Dimensional data and is able to plot it in many different ways.
    """

    def __init__(self, x: ArrayLike[Real], y: ArrayLike[Real]) -> None:
        """
        Parameters
        ----------
        x : ArrayLike[Real]
            1-Dimensional Data for the x-axis
        y : ArrayLike[Real]
            1-Dimensional Data for the y-axis

        Raises
        ------
        ValueError
            If any of the arrays are empty.
        ValueError
            If the arrays are not the same length.
        """
        NDArrayErrors.ARR2D(x, y)
        self._points: np.ndarray = np.column_stack(cast(Sequence[ArrayLike], (x, y)))
        self._original_x: str = f"{x!r}"
        self._original_y: str = f"{y!r}"
        self._fig: Figure | None = None
        self._ax: Axes | None = None

    def __repr__(self) -> str:
        return f"Array2D(x={self._original_x}, y={self._original_y})"

    def __str__(self) -> str:
        return f"Array2D({self._original_x}, {self._original_y})"

    def __hash__(self) -> int:
        return hash(tuple(self._points.tolist()))

    def array_equal(self, other: ArrayLike) -> bool:
        if not isinstance(other, self.__class__):
            return np.array_equal(self._points, other)
        return np.array_equal(self._points, other._points)

    def __eq__(self, other: Any) -> Any:
        if not isinstance(other, self.__class__):
            return self._points == other
        return self._points == other._points

    def __ne__(self, other: Any) -> Any:
        if not isinstance(other, self.__class__):
            return self._points == other
        return self._points != other._points

    def __gt__(self, other: Array2D) -> Any:
        if not isinstance(other, self.__class__):
            return self._points > other
        return self._points > other._points

    def __ge__(self, other: Array2D) -> Any:
        if not isinstance(other, self.__class__):
            return self._points >= other
        return self._points >= other._points

    def __lt__(self, other: Array2D) -> Any:
        if not isinstance(other, self.__class__):
            return self._points < other
        return self._points < other._points

    def __le__(self, other: Array2D) -> Any:
        if not isinstance(other, self.__class__):
            return self._points <= other
        return self._points <= other._points

    @property
    def x(self) -> np.ndarray:
        """1-Dimensional Data for the x-axis"""
        return self._points[:, 0]

    @property
    def y(self) -> np.ndarray:
        """1-Dimensional Data for the y-axis"""
        return self._points[:, 1]

    @property
    def data(self) -> np.ndarray:
        return self._points

    def plot(self) -> list[Line2D]:
        """Creates a line plot of the data.

        Returns
        -------
        list[Line2D]
            The plotted figure.
        """
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots()

        self._fig = cast(Figure, self._fig)
        self._ax = cast(Axes, self._ax)
        return self._ax.plot(self.x, self.y)

    def scatter(
        self,
        s: ArrayLike | float | None = None,
        c: np.typing.ArrayLike | ColorType | Sequence[ColorType] | None = None,
        marker: MarkerType | None = None,
        cmap: str | Colormap | None = None,
        alpha: float | None = None,
    ) -> PathCollection:
        """Creates a scatter plot of the data.

        Parameters
        ----------
        s : ArrayLike[Real] | float | None, optional, default=None
            Sizes of the points.
        c : np.typing.ArrayLike | ColorType | Sequence[ColorType] | None, optional, default=None
            Colors of the points.
        marker : MarkerType | None, optional, default=None
            The shape of the marker.
        cmap: str | Colormap | None, optional, default=None
            Colormap of the points.
        alpha : float | None, optional, default=None
            Alpha of the points.

        Returns
        -------
        PathCollection
            The plotted figure.
        """
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots()

        self._fig = cast(Figure, self._fig)
        self._ax = cast(Axes, self._ax)
        return self._ax.scatter(
                self.x, self.y, s, c, marker=marker, alpha=alpha, cmap=cmap
            )

    def bar(self) -> BarContainer:
        """Creates a bar plot of the data.

        Returns
        -------
        BarContainer
            The plotted figure.
        """
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots()

        self._fig = cast(Figure, self._fig)
        self._ax = cast(Axes, self._ax)
        return self._ax.bar(self.x, self.y)

    def spline(self) -> list[Line2D]:
        """Creates a spline chart of the data.

        Returns
        -------
        list[Line2D]
            The plotted figure.
        """
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots()

        self._fig = cast(Figure, self._fig)
        self._ax = cast(Axes, self._ax)
        CS: Final[CubicSpline[np.float64]] = CubicSpline(self.x, self.y)
        X: Final[np.ndarray] = np.linspace(self.x.min(), self.x.max(), 250)
        Y: Final[np.ndarray] = CS(X)
        return self._ax.plot(X, Y)

    def show(self) -> None:
        """Shows the current figure."""
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots()

        self._fig.show()

    def save(
        self,
        dir: str | Path | None = None,
        suffix: str = "svg",
        *,
        transparent: bool | None = None,
    ) -> None:
        """Saves the current figure to a file.

        Parameters
        ----------
        dir : str | Path | None, optional, default=None
            The path that the figure will be saved.
        suffix : str, optional, default='svg'
            The suffix for the file.
        transparent : bool | None, optional, default=None
            Whether the image is transparent.

        Raises
        ------
        ValueError
            Suffix is not supported.
        """
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots()

        self._fig = cast(Figure, self._fig)
        self._ax = cast(Axes, self._ax)
        if suffix not in _FIGURE_FORMATS:
            raise ValueError(
                f"Format: '{suffix}' is not supported, (supported formats: {', '.join(_FIGURE_FORMATS)})"
            )
        number: int = 1
        filename: str = f"figure_{number}.{suffix}"
        if dir is None:
            dir = Path("figures")
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
            self._fig.savefig(path, transparent=transparent)


class Array1D:
    """
    Array1D
    -------

    Takes in 1-Dimensional data and is able to plot it in many different ways.
    """

    def __init__(self, data: ArrayLike[Real]) -> None:
        """
        Parameters
        ----------
        data : ArrayLike[Real]
            1-Dimensional Data

        Raises
        ------
        ValueError
            If any of the arrays are empty.
        """
        NDArrayErrors.ARR1D(data)
        self._data: np.ndarray = np.array(data)
        self._original: str = f"{data!r}"
        self._fig: Figure | None
        self._ax: Axes | None
        self._fig, self._ax = None, None

    def __repr__(self) -> str:
        return f"Array1D(data={self._original})"

    def __str__(self) -> str:
        return f"Array1D({self._original})"

    def __hash__(self) -> int:
        return hash(tuple(self._data.tolist()))

    def array_equal(self, other: ArrayLike) -> bool:
        if not isinstance(other, self.__class__):
            return np.array_equal(self._data, other)
        return np.array_equal(self._data, other._data)

    def __eq__(self, other: Any) -> Any:
        if not isinstance(other, self.__class__):
            return self._data == other
        return self._data == other._data

    def __ne__(self, other: Any) -> Any:
        if not isinstance(other, self.__class__):
            return self._data == other
        return self._data != other._data

    def __gt__(self, other: Any) -> Any:
        if not isinstance(other, self.__class__):
            return self._data > other
        return self._data > other._data

    def __ge__(self, other: Array1D) -> Any:
        if not isinstance(other, self.__class__):
            return self._data >= other
        return self._data >= other._data

    def __lt__(self, other: Any) -> Any:
        if not isinstance(other, self.__class__):
            return self._data < other
        return self._data < other._data

    def __le__(self, other: Any) -> Any:
        if not isinstance(other, self.__class__):
            return self._data <= other
        return self._data <= other._data

    @property
    def data(self) -> np.ndarray:
        """1-Dimensional Data for the x-axis"""
        return self._data

    def plot(self) -> list[Line2D]:
        """Creates a line plot of the data.

        Returns
        -------
        list[Line2D]
            The plotted figure.
        """
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots()

        self._fig = cast(Figure, self._fig)
        self._ax = cast(Axes, self._ax)
        return self._ax.plot(self._data)

    def bar(self) -> BarContainer:
        """Creates a bar plot of the data.

        Returns
        -------
        BarContainer
            The plotted figure.
        """
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots()

        self._fig = cast(Figure, self._fig)
        self._ax = cast(Axes, self._ax)
        return self._ax.bar(range(len(self._data)), self._data)

    def boxplot(self) -> dict[str, Any]:
        """Creates a boxplot of the data.

        Returns
        -------
        dict[str, Any]
            The plotted figure.
        """
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots()

        self._fig = cast(Figure, self._fig)
        self._ax = cast(Axes, self._ax)
        return self._ax.boxplot(self._data)

    def show(self) -> None:
        """Shows the current figure."""
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots()

        self._fig.show()

    def save(
        self,
        dir: str | Path | None = None,
        suffix: str = "svg",
        *,
        transparent: bool | None = None,
    ) -> None:
        """Saves the current figure to a file.

        Parameters
        ----------
        dir : str | Path | None, optional, default=None
            The path that the figure will be saved.
        suffix : str, optional, default='svg'
            The suffix for the file.
        transparent : bool | None, optional, default=None
            Whether the image is transparent.

        Raises
        ------
        ValueError
            Suffix is not supported.
        """
        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots()

        self._fig = cast(Figure, self._fig)
        self._ax = cast(Axes, self._ax)
        if suffix not in _FIGURE_FORMATS:
            raise ValueError(
                f"Format: '{suffix}' is not supported, (supported formats: {', '.join(_FIGURE_FORMATS)})"
            )
        number: int = 1
        filename: str = f"figure_{number}.{suffix}"
        if dir is None:
            dir = Path("figures")
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
            self._fig.savefig(path, transparent=transparent)

    @property
    def mean(self) -> Real:
        return self._data.mean()

    @property
    def avg(self) -> Real:
        return self.mean

    @property
    def median(self) -> np.floating[Any]:
        return np.median(self._data)

    @property
    def mode(self) -> "ModeResult":
        return stats.mode(self._data)

    @property
    def std(self) -> Real:
        return self.data.std()

    @property
    def stdev(self) -> Real:
        return self.data.std()

    @property
    def quantiles(self) -> np.ndarray:
        quantiles: np.ndarray = np.quantile(self._data, [0.25, 0.5, 0.75])
        if np.iscomplexobj(quantiles):
            raise Exception(
                "This will never happen because NumPy already checks for this"
            )
        return quantiles

    @property
    def q1(self) -> Real:
        return cast(Real, self.quantiles[0])

    @property
    def q3(self) -> Real:
        return cast(Real, self.quantiles[2])

    @property
    def iqr(self) -> Real:
        return cast(Real, self.q3 - self.q1)
