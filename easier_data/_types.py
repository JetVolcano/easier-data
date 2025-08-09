from collections import deque
from collections.abc import Iterable, Sequence
from typing import Any, Final, Generic, TypeVar, overload

import numpy as np


T = TypeVar("T")


class ArrayLike(Generic[T], Sequence[T]):
    def __init__(self, data: Sequence[T]) -> None:
        self._type: Final[type] = type(data)
        self.__data: np.ndarray = np.array(data)

    @property
    def data(self) -> np.ndarray:
        """
        Returns the data of the ArrayLike

        :returntype np.ndarray:
        """
        return self.__data

    def __len__(self) -> int:
        return len(self.__data)

    def __repr__(self) -> str:
        return f"ArrayLike({self._type(self.__data)!r})"

    def __instancecheck__(self, instance) -> Any:
        return super().__instancecheck__(instance)

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> "ArrayLike[T]": ...

    def __getitem__(self, index: int | slice) -> T | "ArrayLike[T]":
        if isinstance(index, slice):
            return ArrayLike(self.__data[index])
        return self.__data[index]


ArrayLike.register(list)
ArrayLike.register(tuple)
ArrayLike.register(deque)
ArrayLike.register(np.ndarray)


def check_type(iterable: Iterable[Any], _type: type | tuple[type, ...]) -> bool:
    """
    Takes a given iterable and checks the type of the values inside to see if it matches the _type parameter

    :param iterable: The iterable to check
    :param _type: The type to check the values against
    :type _type: type | tuple[type, ...]
    :type iterable: Iterable
    :returntype bool:
    """
    if not isinstance(iterable, Iterable):
        return False
    return all(isinstance(item, _type) for item in iterable)
