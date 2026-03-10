from abc import ABC, abstractmethod

from custom_warnings import Unstable


@Unstable("Style is not yet implemented and does nothing")
class Style(ABC):
    def __init__(self) -> None: ...

    @property
    @abstractmethod
    def title(self) -> str: ...

    @title.setter
    @abstractmethod
    def title(self, value: str) -> None: ...


# rough sketch:
# class Array(Style):
#   def __init__(self, ...):
#       self._title = ""
# Array.title = "Hello World"
