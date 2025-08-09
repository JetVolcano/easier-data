from collections.abc import Iterable
from numbers import Real
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.pyplot import show

from .arrays import Array1D, Array2D


_EXPECTED_1D_ARRAY_LEN: Final[int] = 1
_EXPECTED_2D_ARRAY_LEN: Final[int] = 2


def plot(
    array: Iterable[Real] | tuple[Iterable[Real], Iterable[Real]] | Array1D | Array2D,
) -> list[Line2D]:
    if isinstance(array, Array1D | Array2D):
        return array.plot()
    if (
        len(array) == _EXPECTED_2D_ARRAY_LEN
        and isinstance(array[0], Iterable)
        and isinstance(array[1], Iterable)
    ):
        return Array2D(*array).plot()
    if len(array) == _EXPECTED_1D_ARRAY_LEN and isinstance(array[0], Iterable):
        return Array1D(*array).plot()
    return plt.plot(array)


__all__: list[str] = ["show"]
