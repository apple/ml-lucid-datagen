#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from typing import Optional

from pydantic import BaseModel

from lucid_generate_data.utils.definitions import ActionResult, AppContext, RecommendedAction


class Entity(BaseModel):
    def perform(self, app_context: AppContext) -> ActionResult:
        return None

    def recommend_action(self) -> Optional[RecommendedAction]:
        return None


class MutableEntity(Entity):
    def persist(self) -> None:
        """In the future use this to persist this to a database.

        Returns: database ID
        """
        pass

    def __setattr__(self, name: str, value: any) -> None:
        self.__dict__[name] = value
        self.persist()

    class Config:
        allow_mutation = True
