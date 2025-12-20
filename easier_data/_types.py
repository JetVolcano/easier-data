from collections.abc import Iterable, Sequence
from typing import Any, Final, Type, TypeVar

import numpy as np


T_co = TypeVar("T_co", covariant=True)


type ArrayLike[T_co] = Sequence[T_co] | np.ndarray

type INT_TYPES = int | np.integer
type FLOAT_TYPES = float | np.floating
type COMPLEX_TYPES = complex | np.complexfloating

type Real = INT_TYPES | FLOAT_TYPES
type Number = INT_TYPES | FLOAT_TYPES | COMPLEX_TYPES

_INSTANCE_CHECK_REAL: Final[tuple[Type, Type, Type, Type]] = (
    int,
    float,
    np.integer,
    np.floating,
)
_INSTANCE_CHECK_NUMBER: Final[tuple[Type, Type, Type, Type, Type, Type]] = (
    int,
    float,
    complex,
    np.integer,
    np.floating,
    np.complexfloating,
)


def _check_type(iterable: Iterable[Any], _type: Type | tuple[Type, ...]) -> bool:
    """Takes a given iterable and checks the type of the values inside to see if it matches the _type parameter

    Parameters
    ----------
    iterable : Iterable[Any] | Any
        The iterable to check the type of
    _type : type | tuple[type, ...]
        The type to check the values against

    Returns
    -------
    bool
        Whether the check passed
    """
    if not isinstance(iterable, Iterable):
        raise TypeError(f"iterable must be iterable, not: {type(iterable)}")
    return all(isinstance(item, _type) for item in iterable)


__all__: list[str] = [
    "_INSTANCE_CHECK_NUMBER",
    "_INSTANCE_CHECK_REAL",
    "ArrayLike",
    "Number",
    "Real",
    "_check_type", 
]
