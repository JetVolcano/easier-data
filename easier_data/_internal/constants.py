from typing import Any


class Constant(type):
    def __setattr__(cls, name: str, value: Any) -> None:
        if name in cls.__dict__:
            raise AttributeError(f"Cannot modify constant: '{name}'")
        super().__setattr__(name, value)
