"""
Easier Data
-----------

This is a python package for managing data and plotting easier
"""

from . import plotting
from .arrays import Array1D, Array2D, Array3D
from .stats import ComplexDataSet, DataSet


__all__: list[str] = [
    "Array1D",
    "Array2D",
    "Array3D",
    "ComplexDataSet",
    "DataSet",
    "arrays",
    "plotting",
    "stats",
]

__version__: str = "1.0.0"
