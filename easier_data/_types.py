from collections import deque
from collections.abc import Sequence
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
        Data of the ArrayLike object
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


def _check_type(iterable: ArrayLike[Any], _type: type | tuple[type, ...]) -> bool:
    """Takes a given iterable and checks the type of the values inside to see if it matches the _type parameter

    Parameters
    ----------
    iterable : ArrayLike[Any]
        The iterable to check the type of
    _type : type | tuple[type, ...]
        The type to check the values against

    Returns
    -------
    bool
        Whether the check passed
    """
    if not isinstance(iterable, ArrayLike):
        return False
    return all(isinstance(item, _type) for item in iterable)


__all__: list[str] = [
    "ArrayLike",
    "_check_type",
]
