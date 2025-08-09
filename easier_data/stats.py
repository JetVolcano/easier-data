from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import partial
from numbers import Number, Real
from typing import TYPE_CHECKING, Any, ClassVar, Final

import numpy as np
from scipy import stats

from ._types import ArrayLike, check_type


if TYPE_CHECKING:
    from scipy.stats._stats_py import ModeResult


class DataSet:
    __hash__: ClassVar[None] = None

    def __init__(self, data: ArrayLike[Real]) -> None:
        if len(data) == 0:
            raise ValueError("Data cannot be empty.")
        if not check_type(data, Real):
            raise TypeError("Data must only contain real numbers.")
        self.__data: np.ndarray = np.array(data)
        self._type: type = type(data)
        self.trim_mean: Final[Callable[[float, int | None], Real]] = partial(
            stats.trim_mean, self.__data
        )

    def __repr__(self) -> str:
        """
        Returns the representation of the DataSet

        :returntype str:
        """
        return f"DataSet(data={self._type(self.__data) if self._type != np.ndarray else self.__data!r})"

    def __str__(self) -> str:
        """
        Returns the string representation of the DataSet

        :returntype str:
        """
        return f"DataSet({self._type(self.__data) if self._type != np.ndarray else self.__data!r})"

    def __eq__(self, other: DataSet) -> Any:
        return self.__data == other.__data

    def __ne__(self, other: DataSet) -> Any:
        return self.__data != other.__data

    def __gt__(self, other: DataSet) -> Any:
        return self.__data > other.__data

    def __ge__(self, other: DataSet) -> Any:
        return self.__data >= other.__data

    def __lt__(self, other: DataSet) -> Any:
        return self.__data < other.__data

    def __le__(self, other: DataSet) -> Any:
        return self.__data <= other.__data

    @property
    def data(self) -> np.ndarray:
        """
        Returns the data of the DataSet

        :returntype np.ndarray:
        """
        return self.__data

    @property
    def mean(self) -> Real:
        """
        Returns the mean of the DataSet

        :returntype Real:
        """
        return self.__data.mean()

    @property
    def median(self) -> Real:
        """
        Returns the median of the DataSet

        :returntype Real:
        """
        return np.median(self.__data)

    @property
    def mode(self) -> "ModeResult":
        """
        Returns the mode of the DataSet

        :returntype ModeResult:
        """
        return stats.mode(self.__data)

    @property
    def std(self) -> Real:
        """
        Returns the standard deviation of the DataSet

        :returntype Real:
        """
        return self.__data.std()

    @property
    def stdev(self) -> Real:
        """
        Returns the standard deviation of the DataSet

        :returntype Real:
        """
        return self.__data.std()

    @property
    def quantiles(self) -> np.ndarray:
        """
        Returns the quantiles of the DataSet

        :returntype np.ndarray:
        """
        return np.quantile(self.__data, [0.25, 0.5, 0.75])

    @property
    def q1(self) -> Real:
        """
        Returns the first quartile of the DataSet

        :returntype Real:
        """
        return self.quantiles[0]

    @property
    def q3(self) -> Real:
        """
        Returns the third quartile of the DataSet

        :returntype Real:
        """
        return self.quantiles[2]

    @property
    def iqr(self) -> Real:
        """
        Returns the interquartile range of the DataSet

        :returntype Real:
        """
        return self.q3 - self.q1

    def append(self, obj: Real) -> None:
        """
        Appends an object to the DataSet

        :param obj: The object to append
        """
        self.__data = np.append(self.__data, obj)

    def appendleft(self, x: Real) -> None:
        """
        Appends an object to the left side of the DataSet

        :param x: The object to append
        """
        self.__data = np.insert(self.__data, 0, x)

    def extend(self, iterable: Iterable[Real]) -> None:
        """
        Extends an iterable to the DataSet

        :param iterable: The iterable to extend
        """
        self.__data = np.concatenate((self.__data, iterable))

    def extendleft(self, iterable: Iterable[Real]) -> None:
        """
        Extends an iterable to the left side of the DataSet

        :param iterable: The iterable to extend
        """
        self.__data = np.concatenate((np.array(iterable)[::-1], self.__data))


class ComplexDataSet:
    __hash__: ClassVar[None] = None

    def __init__(self, data: ArrayLike[Number]) -> None:
        if len(data) == 0:
            raise ValueError("Data cannot be empty.")
        if not check_type(data, Number):
            raise TypeError("Data must contain only complex or real numbers.")
        self.__data: np.ndarray = np.array(data)
        self._type: type = type(data)
        self.trim_mean: Final[Callable[[float, int | None], Number]] = partial(
            stats.trim_mean, self.__data
        )

    def __repr__(self) -> str:
        """
        Returns the representation of the ComplexDataSet

        :returntype str:
        """
        return f"ComplexDataSet(data={self._type(self.__data) if self._type != np.ndarray else self.__data!r})"

    def __str__(self) -> str:
        """
        Returns the string representation of the ComplexDataSet

        :returntype str:
        """
        return f"ComplexDataSet({self._type(self.__data) if self._type != np.ndarray else self.__data!r})"

    def __eq__(self, other: ComplexDataSet) -> Any:
        return self.__data == other.__data

    def __ne__(self, other: ComplexDataSet) -> Any:
        return self.__data != other.__data

    def __gt__(self, other: ComplexDataSet) -> Any:
        return self.__data > other.__data

    def __ge__(self, other: ComplexDataSet) -> Any:
        return self.__data >= other.__data

    def __lt__(self, other: ComplexDataSet) -> Any:
        return self.__data < other.__data

    def __le__(self, other: ComplexDataSet) -> Any:
        return self.__data <= other.__data

    @property
    def data(self) -> np.ndarray:
        """
        Returns the data of the DataSet

        :returntype np.ndarray:
        """
        return self.__data

    @property
    def mean(self) -> Number:
        """
        Returns the mean of the DataSet

        :returntype Number:
        """
        return self.__data.mean()

    @property
    def median(self) -> Number:
        """
        Returns the median of the DataSet

        :returntype Number:
        """
        return np.median(self.__data)

    @property
    def mode(self) -> "ModeResult":
        """
        Returns the mode of the DataSet

        :returntype ModeResult:
        """
        return stats.mode(self.__data)

    @property
    def std(self) -> Number:
        """
        Returns the standard deviation of the DataSet

        :returntype Number:
        """
        return self.__data.std()

    @property
    def stdev(self) -> Number:
        """
        Returns the standard deviation of the DataSet

        :returntype Number:
        """
        return self.__data.std()

    def append(self, obj: Number) -> None:
        """
        Appends an object to the DataSet

        :param obj: The object to append
        """
        self.__data = np.append(self.__data, obj)

    def appendleft(self, x: Number) -> None:
        """
        Appends an object to the left side of the DataSet

        :param x: The object to append
        """
        self.__data = np.insert(self.__data, 0, x)

    def extend(self, iterable: Iterable[Number]) -> None:
        """
        Extends an iterable to the DataSet

        :param iterable: The iterable to extend
        """
        self.__data = np.concatenate((self.__data, iterable))

    def extendleft(self, iterable: Iterable[Number]) -> None:
        """
        Extends an iterable to the left side of the DataSet

        :param iterable: The iterable to extend
        """
        self.__data = np.concatenate((np.array(iterable)[::-1], self.__data))
