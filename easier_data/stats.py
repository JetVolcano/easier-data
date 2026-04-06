from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
from scipy import stats

from ._internal._types import (
    _INSTANCE_CHECK_NUMBER,
    _INSTANCE_CHECK_REAL,
    ArrayLike,
    Number,
    Real,
    _check_type,
)
from ._internal.errors import DataSetErrors


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
            If any of the arrays are empty.
        """
        DataSetErrors.DS(data)
        self._data: np.ndarray = np.array(data)
        self._orginal: str = f"{data!r}"
        self.trim_mean: Final[Callable[..., _FloatOrND]] = partial(
            stats.trim_mean, self._data
        )

    def __repr__(self) -> str:
        return f"DataSet(data={self._orginal})"

    def __str__(self) -> str:
        return f"DataSet({self._orginal})"

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

    def __ge__(self, other: Any) -> Any:
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
        return self._data

    @property
    def mean(self) -> Real:
        return self._data.mean()

    @property
    def median(self) -> Real:
        return cast(Real, np.median(self._data))

    @property
    def mode(self) -> "ModeResult":
        return stats.mode(self._data)

    @property
    def std(self) -> Real:
        return self._data.std()

    @property
    def stdev(self) -> Real:
        return self._data.std()

    @property
    def quantiles(self) -> np.ndarray:
        return np.quantile(self._data, [0.25, 0.5, 0.75])

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
        self._data = np.append(self._data, np.array([obj]))

    def appendleft(self, x: Real) -> None:
        """Appends an object to the left side of the DataSet.

        Parameters
        ----------
        x : Real
            The object to append.
        """
        if not isinstance(x, _INSTANCE_CHECK_REAL):
            raise TypeError(f"x must be a real number. not type: {type(x)}.")
        self._data = np.insert(self._data, 0, np.array([x]))

    def extend(self, iterable: Iterable[Real]) -> None:
        """Extends an iterable to the DataSet.

        Parameters
        ----------
        iterable : Iterable[Real]
            The iterable to extend.
        """
        if not _check_type(iterable, _INSTANCE_CHECK_REAL):
            raise TypeError("iterable must only contain real numbers.")
        self._data = np.concatenate((self._data, np.array(iterable)))

    def extendleft(self, iterable: Iterable[Real]) -> None:
        """Extends an iterable to the left side of the DataSet.

        Parameters
        ----------
        iterable : Iterable[Real]
            The iterable to extend.
        """
        if not _check_type(iterable, _INSTANCE_CHECK_REAL):
            raise TypeError("iterable must only contain real numbers.")
        self._data = np.concatenate((np.array(iterable)[::-1], self._data))


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
            If any of the arrays are empty.
        """
        DataSetErrors.CDS(data)
        self._data: np.ndarray = np.array(data)
        self._orginal: str = f"{data!r}"
        self.trim_mean: Final[Callable[..., _FloatOrND]] = partial(
            stats.trim_mean, self._data
        )

    def __repr__(self) -> str:
        return f"ComplexDataSet(data={self._orginal})"

    def __str__(self) -> str:
        return f"ComplexDataSet({self._orginal})"

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

    def __ge__(self, other: Any) -> Any:
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
        return self._data

    @property
    def mean(self) -> Number:
        return self._data.mean()

    @property
    def median(self) -> Number:
        return cast(Number, np.median(self._data))

    @property
    def mode(self) -> "ModeResult":
        return stats.mode(self._data)

    @property
    def std(self) -> Number:
        return self._data.std()

    @property
    def stdev(self) -> Number:
        return self._data.std()

    def append(self, obj: Number) -> None:
        """Appends an object to the DataSet.

        Parameters
        ---------
        obj : Number
            The object to append.
        """
        if not isinstance(obj, _INSTANCE_CHECK_NUMBER):
            raise TypeError(f"obj must be a number not type: {type(obj)}.")
        self._data = np.append(self._data, np.array([obj]))

    def appendleft(self, x: Number) -> None:
        """Appends an object to the left side of the DataSet.

        Parameters
        ----------
        x : Number
            The object to append.
        """
        if not isinstance(x, _INSTANCE_CHECK_NUMBER):
            raise TypeError(f"x must be a number not type: {type(x)}.")
        self._data = np.insert(self._data, 0, np.array([x]))

    def extend(self, iterable: Iterable[Number]) -> None:
        """Extends an iterable to the DataSet.

        Parameters
        ----------
        iterable : Iterable[Number]
            The iterable to extend.
        """
        if not _check_type(iterable, _INSTANCE_CHECK_NUMBER):
            raise TypeError("iterable must only contain numbers.")
        self._data = np.concatenate((self._data, np.array(iterable)))

    def extendleft(self, iterable: Iterable[Number]) -> None:
        """Extends an iterable to the left side of the DataSet.

        Parameters
        ----------
        iterable : Iterable[Real]
            The iterable to extend.
        """
        if not _check_type(iterable, _INSTANCE_CHECK_NUMBER):
            raise TypeError("iterable must only contain numbers.")
        self._data = np.concatenate((np.array(iterable)[::-1], self._data))
