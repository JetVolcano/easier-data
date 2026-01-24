from collections.abc import Sequence
from typing import Final

from ._types import (
    _INSTANCE_CHECK_ARRAYLIKE,
    _INSTANCE_CHECK_NUMBER,
    _INSTANCE_CHECK_REAL,
    ArrayLike,
    Number,
    Real,
    _check_type,
)


class NDArrayErrors:
    ARR3D_ERRORS: Final[Sequence[TypeError | ValueError]] = [
        TypeError("x must be of type ArrayLike[Real]"),
        TypeError("y must be of type ArrayLike[Real]"),
        TypeError("z must be of type ArrayLike[Real]"),
        TypeError("x must contain only real numbers"),
        TypeError("y must contain only real numbers"),
        TypeError("z must contain only real numbers"),
        ValueError("Data cannot be empty."),
        ValueError("x, y, and z must be the same length."),
    ]
    ARR2D_ERRORS: Final[Sequence[TypeError | ValueError]] = [
        TypeError("x must be of type ArrayLike[Real]"),
        TypeError("y must be of type ArrayLike[Real]"),
        TypeError("x must contain only real numbers"),
        TypeError("y must contain only real numbers"),
        ValueError("Data cannot be empty."),
        ValueError("x and y must be the same length."),
    ]
    ARR1D_ERRORS: Final[Sequence[TypeError | ValueError]] = [
        TypeError("Data must be of type ArrayLike[Real]"),
        TypeError("Data must contain only real numbers"),
        ValueError("Data cannot be empty."),
    ]

    @staticmethod
    def ARR3D(x: ArrayLike[Real], y: ArrayLike[Real], z: ArrayLike[Real]) -> None:
        EG_ERRORS: Final[Sequence[ValueError | TypeError]] = []

        def _check(index: int, arr: ArrayLike[Real]) -> None:
            if isinstance(arr, _INSTANCE_CHECK_ARRAYLIKE):
                if not _check_type(arr, _INSTANCE_CHECK_REAL):
                    EG_ERRORS.append(NDArrayErrors.ARR3D_ERRORS[3 + index])
            else:
                EG_ERRORS.append(NDArrayErrors.ARR3D_ERRORS[index])

        _check(0, x)
        _check(1, y)
        _check(2, z)

        if all(isinstance(arr, _INSTANCE_CHECK_ARRAYLIKE) for arr in (x, y, z)):
            if any(len(arr) == 0 for arr in (x, y, z)):
                EG_ERRORS.append(NDArrayErrors.ARR3D_ERRORS[6])
            if len(x) != len(y) or len(y) != len(z):
                EG_ERRORS.append(NDArrayErrors.ARR3D_ERRORS[7])

        if EG_ERRORS:
            raise ExceptionGroup(f"{len(EG_ERRORS)} Error(s) occurred", EG_ERRORS)

    @staticmethod
    def ARR2D(x: ArrayLike[Real], y: ArrayLike[Real]) -> None:
        EG_ERRORS: Final[Sequence[ValueError | TypeError]] = []

        def _check(index: int, arr: ArrayLike[Real]) -> None:
            if isinstance(arr, _INSTANCE_CHECK_ARRAYLIKE):
                if not _check_type(arr, _INSTANCE_CHECK_REAL):
                    EG_ERRORS.append(NDArrayErrors.ARR2D_ERRORS[2 + index])
            else:
                EG_ERRORS.append(NDArrayErrors.ARR2D_ERRORS[index])

        _check(0, x)
        _check(1, y)

        if all(isinstance(arr, _INSTANCE_CHECK_ARRAYLIKE) for arr in (x, y)):
            if any(len(arr) == 0 for arr in (x, y)):
                EG_ERRORS.append(NDArrayErrors.ARR2D_ERRORS[4])
            if len(x) != len(y):
                EG_ERRORS.append(NDArrayErrors.ARR2D_ERRORS[5])

        if EG_ERRORS:
            raise ExceptionGroup(f"{len(EG_ERRORS)} Error(s) occurred", EG_ERRORS)

    @staticmethod
    def ARR1D(x: ArrayLike[Real]) -> None:
        EG_ERRORS: Final[Sequence[ValueError | TypeError]] = []

        def _check(index: int, arr: ArrayLike[Real]) -> None:
            if isinstance(arr, _INSTANCE_CHECK_ARRAYLIKE):
                if not _check_type(arr, _INSTANCE_CHECK_REAL):
                    EG_ERRORS.append(NDArrayErrors.ARR1D_ERRORS[1 + index])
            else:
                EG_ERRORS.append(NDArrayErrors.ARR1D_ERRORS[index])

        _check(0, x)

        if isinstance(x, _INSTANCE_CHECK_ARRAYLIKE):
            if len(x) == 0:
                EG_ERRORS.append(NDArrayErrors.ARR1D_ERRORS[2])

        if EG_ERRORS:
            raise ExceptionGroup(f"{len(EG_ERRORS)} Error(s) occurred", EG_ERRORS)


class DataSetErrors:
    DS_ERRORS: Final[Sequence[TypeError | ValueError]] = [
        TypeError("Data must be of type ArrayLike[Real]"),
        TypeError("Data must contain only real numbers"),
        ValueError("Data cannot be empty."),
    ]
    CDS_ERRORS: Final[Sequence[TypeError | ValueError]] = [
        TypeError("Data must be of type ArrayLike[Number]"),
        TypeError("Data must contain only numbers"),
        ValueError("Data cannot be empty."),
    ]

    @staticmethod
    def DS(data: ArrayLike[Real]) -> None:
        EG_ERRORS: Final[Sequence[ValueError | TypeError]] = []

        def _check(index: int, arr: ArrayLike[Real]) -> None:
            if isinstance(arr, _INSTANCE_CHECK_ARRAYLIKE):
                if not _check_type(arr, _INSTANCE_CHECK_REAL):
                    EG_ERRORS.append(DataSetErrors.DS_ERRORS[1 + index])
            else:
                EG_ERRORS.append(DataSetErrors.DS_ERRORS[index])

        _check(0, data)

        if isinstance(data, _INSTANCE_CHECK_ARRAYLIKE):
            if len(data) == 0:
                EG_ERRORS.append(DataSetErrors.DS_ERRORS[2])

        if EG_ERRORS:
            raise ExceptionGroup(f"{len(EG_ERRORS)} Error(s) occurred", EG_ERRORS)

    @staticmethod
    def CDS(data: ArrayLike[Number]) -> None:
        EG_ERRORS: Final[Sequence[ValueError | TypeError]] = []

        def _check(index: int, arr: ArrayLike[Number]) -> None:
            if isinstance(arr, _INSTANCE_CHECK_ARRAYLIKE):
                if not _check_type(arr, _INSTANCE_CHECK_NUMBER):
                    EG_ERRORS.append(DataSetErrors.CDS_ERRORS[1 + index])
            else:
                EG_ERRORS.append(DataSetErrors.CDS_ERRORS[index])

        _check(0, data)

        if isinstance(data, _INSTANCE_CHECK_ARRAYLIKE):
            if len(data) == 0:
                EG_ERRORS.append(DataSetErrors.CDS_ERRORS[2])

        if EG_ERRORS:
            raise ExceptionGroup(f"{len(EG_ERRORS)} Error(s) occurred", EG_ERRORS)
