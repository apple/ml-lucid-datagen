#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from abc import ABC, abstractmethod
from typing import Any, Dict, Type


class Stage(ABC):
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        register_class(cls)

    @abstractmethod
    def __call__(self, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError


_registry: Dict[str, Type[Stage]] = {}


def register_class(cls: Type[Stage]) -> None:
    _registry[cls.__name__] = cls


def stage_factory(name: str) -> Stage:
    return _registry[name]()


class StageExecutionException(Exception):
    pass
