from collections import deque
from collections.abc import Sequence
from typing import Final, Generic, TypeVar, overload

import numpy as np


T = TypeVar("T")


class ArrayLike(Generic[T], Sequence[T]):
    def __init__(self, data: Sequence[T]) -> None:
        self._type: Final[type] = type(data)
        self.__data: np.ndarray = np.array(data)

    def __len__(self) -> int:
        return len(self.__data)

    def __repr__(self) -> str:
        return f"ArrayLike({self._type(self.__data)!r})"

    def __instancecheck__(self, instance):
        return super().__instancecheck__(instance)

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> "ArrayLike[T]": ...

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return ArrayLike(self.__data[index])
        return self.__data[index]


ArrayLike.register(list)
ArrayLike.register(tuple)
ArrayLike.register(deque)
ArrayLike.register(np.ndarray)
