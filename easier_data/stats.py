from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
from scipy import stats

from ._types import (
    _INSTANCE_CHECK_NUMBER,
    _INSTANCE_CHECK_REAL,
    ArrayLike,
    Number,
    Real,
    _check_type,
)


if TYPE_CHECKING:
    from scipy.stats._stats_py import ModeResult, _FloatOrND


class DataSet:
    def __init__(self, data: ArrayLike[Real]) -> None:
        """
        Parameters
        ----------
        data : ArrayLike[Real]
            1-Dimensional data for the DataSet.

        Raises
        ------
        ValueError
            Data cannot be empty.
        TypeError
            Data must only contain real numbers.
        """
        if len(data) == 0:
            raise ValueError("Data cannot be empty.")
        if not _check_type(data, _INSTANCE_CHECK_REAL):
            raise TypeError("Data must only contain real numbers.")
        self.__data: np.ndarray = np.array(data)
        self.__orginal: str = f"{data!r}"
        self.trim_mean: Final[Callable[..., _FloatOrND]] = partial(
            stats.trim_mean, self.__data
        )

    def __repr__(self) -> str:
        return f"DataSet(data={self.__orginal})"

    def __str__(self) -> str:
        return f"DataSet({self.__orginal})"

    def __hash__(self) -> int:
        return hash(tuple(self.__data.tolist()))

    def array_equal(self, other: ArrayLike) -> bool:
        if not isinstance(other, self.__class__):
            return np.array_equal(self.__data, other)
        return np.array_equal(self.__data, other.__data)

    def __eq__(self, other: Any) -> Any:
        if not isinstance(other, self.__class__):
            return self.__data == other
        return self.__data == other.__data

    def __ne__(self, other: Any) -> Any:
        if not isinstance(other, self.__class__):
            return self.__data == other
        return self.__data != other.__data

    def __gt__(self, other: Any) -> Any:
        if not isinstance(other, self.__class__):
            return self.__data > other
        return self.__data > other.__data

    def __ge__(self, other: Any) -> Any:
        if not isinstance(other, self.__class__):
            return self.__data >= other
        return self.__data >= other.__data

    def __lt__(self, other: Any) -> Any:
        if not isinstance(other, self.__class__):
            return self.__data < other
        return self.__data < other.__data

    def __le__(self, other: Any) -> Any:
        if not isinstance(other, self.__class__):
            return self.__data <= other
        return self.__data <= other.__data

    @property
    def data(self) -> np.ndarray:
        return self.__data

    @property
    def mean(self) -> Real:
        return self.__data.mean()

    @property
    def median(self) -> Real:
        return cast(Real, np.median(self.__data))

    @property
    def mode(self) -> "ModeResult":
        return stats.mode(self.__data)

    @property
    def std(self) -> Real:
        return self.__data.std()

    @property
    def stdev(self) -> Real:
        return self.__data.std()

    @property
    def quantiles(self) -> np.ndarray:
        return np.quantile(self.__data, [0.25, 0.5, 0.75])

    @property
    def q1(self) -> Real:
        return self.quantiles[0]

    @property
    def q3(self) -> Real:
        return self.quantiles[2]

    @property
    def iqr(self) -> Real:
        return cast(Real, self.q3 - self.q1)

    def append(self, obj: Real) -> None:
        """Appends an object to the DataSet.

        Parameters
        ---------
        obj : Real
            The object to append.
        """
        if not isinstance(obj, _INSTANCE_CHECK_REAL):
            raise TypeError(f"obj must be a real number. not type: {type(obj)}.")
        self.__data = np.append(self.__data, np.array([obj]))

    def appendleft(self, x: Real) -> None:
        """Appends an object to the left side of the DataSet.

        Parameters
        ----------
        x : Real
            The object to append.
        """
        if not isinstance(x, _INSTANCE_CHECK_REAL):
            raise TypeError(f"x must be a real number. not type: {type(x)}.")
        self.__data = np.insert(self.__data, 0, np.array([x]))

    def extend(self, iterable: Iterable[Real]) -> None:
        """Extends an iterable to the DataSet.

        Parameters
        ----------
        iterable : Iterable[Real]
            The iterable to extend.
        """
        if not _check_type(iterable, _INSTANCE_CHECK_REAL):
            raise TypeError("iterable must only contain real numbers.")
        self.__data = np.concatenate((self.__data, np.array(iterable)))

    def extendleft(self, iterable: Iterable[Real]) -> None:
        """Extends an iterable to the left side of the DataSet.

        Parameters
        ----------
        iterable : Iterable[Real]
            The iterable to extend.
        """
        if not _check_type(iterable, _INSTANCE_CHECK_REAL):
            raise TypeError("iterable must only contain real numbers.")
        self.__data = np.concatenate((np.array(iterable)[::-1], self.__data))


class ComplexDataSet:
    def __init__(self, data: ArrayLike[Number]) -> None:
        """
        Parameters
        ----------
        data : ArrayLike[Number]
            1-Dimensional data for the DataSet.

        Raises
        ------
        ValueError
            Data cannot be empty.
        TypeError
            Data must contain only numbers.
        """
        if len(data) == 0:
            raise ValueError("Data cannot be empty.")
        if not _check_type(data, _INSTANCE_CHECK_NUMBER):
            raise TypeError("Data must contain only numbers.")
        self.__data: np.ndarray = np.array(data)
        self.__orginal: str = f"{data!r}"
        self.trim_mean: Final[Callable[..., _FloatOrND]] = partial(
            stats.trim_mean, self.__data
        )

    def __repr__(self) -> str:
        return f"ComplexDataSet(data={self.__orginal})"

    def __str__(self) -> str:
        return f"ComplexDataSet({self.__orginal})"

    def __hash__(self) -> int:
        return hash(tuple(self.__data.tolist()))

    def array_equal(self, other: ArrayLike) -> bool:
        if not isinstance(other, self.__class__):
            return np.array_equal(self.__data, other)
        return np.array_equal(self.__data, other.__data)

    def __eq__(self, other: Any) -> Any:
        if not isinstance(other, self.__class__):
            return self.__data == other
        return self.__data == other.__data

    def __ne__(self, other: Any) -> Any:
        if not isinstance(other, self.__class__):
            return self.__data == other
        return self.__data != other.__data

    def __gt__(self, other: Any) -> Any:
        if not isinstance(other, self.__class__):
            return self.__data > other
        return self.__data > other.__data

    def __ge__(self, other: Any) -> Any:
        if not isinstance(other, self.__class__):
            return self.__data >= other
        return self.__data >= other.__data

    def __lt__(self, other: Any) -> Any:
        if not isinstance(other, self.__class__):
            return self.__data < other
        return self.__data < other.__data

    def __le__(self, other: Any) -> Any:
        if not isinstance(other, self.__class__):
            return self.__data <= other
        return self.__data <= other.__data

    @property
    def data(self) -> np.ndarray:
        return self.__data

    @property
    def mean(self) -> Number:
        return self.__data.mean()

    @property
    def median(self) -> Number:
        return cast(Number, np.median(self.__data))

    @property
    def mode(self) -> "ModeResult":
        return stats.mode(self.__data)

    @property
    def std(self) -> Number:
        return self.__data.std()

    @property
    def stdev(self) -> Number:
        return self.__data.std()

    def append(self, obj: Number) -> None:
        """Appends an object to the DataSet.

        Parameters
        ---------
        obj : Number
            The object to append.
        """
        if not isinstance(obj, _INSTANCE_CHECK_NUMBER):
            raise TypeError(f"obj must be a number not type: {type(obj)}.")
        self.__data = np.append(self.__data, np.array([obj]))

    def appendleft(self, x: Number) -> None:
        """Appends an object to the left side of the DataSet.

        Parameters
        ----------
        x : Number
            The object to append.
        """
        if not isinstance(x, _INSTANCE_CHECK_NUMBER):
            raise TypeError(f"x must be a number not type: {type(x)}.")
        self.__data = np.insert(self.__data, 0, np.array([x]))

    def extend(self, iterable: Iterable[Number]) -> None:
        """Extends an iterable to the DataSet.

        Parameters
        ----------
        iterable : Iterable[Number]
            The iterable to extend.
        """
        if not _check_type(iterable, _INSTANCE_CHECK_NUMBER):
            raise TypeError("iterable must only contain numbers.")
        self.__data = np.concatenate((self.__data, np.array(iterable)))

    def extendleft(self, iterable: Iterable[Number]) -> None:
        """Extends an iterable to the left side of the DataSet.

        Parameters
        ----------
        iterable : Iterable[Real]
            The iterable to extend.
        """
        if not _check_type(iterable, _INSTANCE_CHECK_NUMBER):
            raise TypeError("iterable must only contain numbers.")
        self.__data = np.concatenate((np.array(iterable)[::-1], self.__data))
