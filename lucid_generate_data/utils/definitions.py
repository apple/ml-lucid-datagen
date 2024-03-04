#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from dataclasses import dataclass
from typing import Any, Optional

AppContext = dict[str, Any]


@dataclass
class Turn:
    pass


@dataclass
class LucidTurn(Turn):
    response: str


@dataclass
class ProgramTurn(Turn):
    index: int
    expression: str
    errors: Optional[str] = None


@dataclass
class AutoTurn(Turn):
    """A turn programmaticaly inserted into the transcript based on the app response."""

    index: int
    expression: str


@dataclass
class AutoTransientTurn(Turn):
    """A turn programmaticaly inserted into the transcript based on the app response.
    It is only showed if it is the last turn.
    """

    index: int
    expression: str


@dataclass
class UserTurn(Turn):
    query: str
    tags: list[str]


@dataclass
class Inform:
    dialogue: str


@dataclass
class InformList(Inform):
    items: list[Any]


@dataclass
class RecommendedAction:
    dialogue: str
    name: str = "anonymous"


class RequestValue(RecommendedAction):
    pass


class RequestDisambiguation(RecommendedAction):
    options: list[str]


class RequestConfirmation(RecommendedAction):
    pass


@dataclass
class ActionResult:
    index: int
    result: Any = None
    recommended_action: Optional[RecommendedAction] = None
