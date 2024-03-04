#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from typing import Any, Dict, List, Optional
from pydantic import create_model

from lucid_generate_data.utils.commands import Command, EntityQuery, camel_to_snake_case
from lucid_generate_data.utils.definitions import (
    AppContext,
    Inform,
    InformList,
    RequestConfirmation,
    RequestValue,
)
from lucid_generate_data.utils.entities import MutableEntity

AppIntent = Dict[str, Any]
AppEntity = Dict[str, Any]


class GenericCommand(Command):
    def recommend_action(self):
        for field in self._required_fields:
            if getattr(self, field) is None:
                return RequestValue(dialogue=f"Ask for {field}")
        if hasattr(self, "confirmed") and not getattr(self, "confirmed"):
            return RequestConfirmation(dialogue="Please confirm")

    def perform(self, app_context: AppContext) -> Inform:
        return Inform("Success")


class GenericEntity(MutableEntity):
    def __str__(self) -> str:
        assert hasattr(self, "entity_name")
        entity_name = getattr(self, "entity_name")
        entity_attributes = vars(self)
        rep = [entity_name, "with"]
        for attr in entity_attributes:
            if attr == "entity_name" or entity_attributes[attr] is None:
                continue
            rep.append(attr)
            rep.append(str(entity_attributes[attr]))
        return " ".join(rep)


class GenericQuery(EntityQuery):
    def perform(self, app_context: AppContext) -> Inform:
        assert hasattr(self, "entity_name")
        entity_name = getattr(self, "entity_name")
        entities = app_context[entity_name]

        self.entities = entities
        assert len(self.entities) == 1
        return InformList(
            f"Found {entity_name}: {self.entities[0]}",
            self.entities,
        )


"""
Convert an intent JSON to an intent class. e.g.,

{
  "command": "track_fitness_activity",
  "args": {
    "activity_type": {
      "type": "str",
      "values": [
        "running",
        "walking",
        "cycling",
      ],
      "optional": false,
      "disambiguation": true
    },
    "duration": {
      "type": "int",
      "values": [
        10,
        15,
        30,
      ],
      "optional": false
    },
  },
  "confirmation_required": false,
  "description": "track daily fitness activities"
}
"""


def intent_to_class(app_intent: AppIntent) -> GenericCommand:
    command = camel_to_snake_case(app_intent["command"])
    fields = {}
    for arg in app_intent["args"]:
        fields[arg] = (Optional[eval(app_intent["args"][arg]["type"])], None)
    if app_intent["confirmation_required"]:
        fields["confirmed"] = (bool, False)
    required_fields = [arg for arg in app_intent["args"] if not app_intent["args"][arg]["optional"]]
    fields["_required_fields"] = (List[str], required_fields)
    commandClass = create_model(
        command,
        __base__=GenericCommand,
        **fields,
    )
    return commandClass


"""
Convert an entity JSON to an entity query. e.g.,

{"entity": "alarm",
 "attributes": {
    "time": {"type": str},
    "label": {"type": str}
 }
}
"""


def entity_to_query_class(app_entity: AppEntity) -> GenericQuery:
    class_name = camel_to_snake_case("find_" + app_entity["entity"])
    fields = {}
    fields["entity_name"] = (str, app_entity["entity"])
    fields["select"] = (Optional[str], None)
    for attr in app_entity["attributes"]:
        fields[attr] = (Optional[app_entity["attributes"][attr]["type"]], None)
    queryClass = create_model(
        class_name,
        __base__=GenericQuery,
        **fields,
    )
    return queryClass


def entity_to_class(app_entity: AppEntity) -> GenericEntity:
    class_name = camel_to_snake_case(app_entity["entity"] + "Entity")
    fields = {}
    fields["entity_name"] = (str, app_entity["entity"])
    for attr in app_entity["attributes"]:
        fields[attr] = (Optional[app_entity["attributes"][attr]["type"]], None)
    entityClass = create_model(
        class_name,
        __base__=GenericEntity,
        **fields,
    )
    return entityClass


def intent_to_func_def(app_intent: AppIntent) -> str:
    arg_strings = []
    for arg in app_intent["args"]:
        arg_strings.append(f"{arg}: {app_intent['args'][arg]['type']}")
    command_definition_string = f"{app_intent['command']}(" + ", ".join(arg_strings) + ")"
    return command_definition_string


def entity_to_query_def(app_entity: AppEntity) -> str:
    arg_strings = []
    for arg in app_entity["attributes"]:
        arg_strings.append(f"{arg}: {app_entity['attributes'][arg]['type']}")
    query_definition_string = f"find_{app_entity['entity']}(" + ", ".join(arg_strings) + ")"

    query_definition_string += f" - Searches for existing {app_entity['entity']} in the database. Use date_of_{app_entity['entity']} for any date information, e.g. date_of_{app_entity['entity']} = 'most recent'"

    return query_definition_string


def create_entity_from_intent(app_intent: AppIntent) -> AppEntity:
    """Convert an app intent JSON to an app entity JSON.
    All arguments of the intent becomes properties of the entity.
    """
    entity_name = app_intent["entity_name"]

    app_entity = {"entity": entity_name, "attributes": {}}
    for arg in app_intent["args"]:
        app_entity["attributes"][arg] = {"type": app_intent["args"][arg]["type"]}
    app_entity["attributes"][f"date_of_{entity_name}"] = {"type": "str"}
    return app_entity


def split_command_into_slots(command):
    split_points = [0]
    brackets_open = False
    for char_no, char in enumerate(command):
        if char == '"':
            brackets_open = not brackets_open
        if not brackets_open and char == ",":
            split_points.append(char_no)

    split_points.append(len(command))

    split = [command[split_points[i] : split_points[i + 1]] for i in range(len(split_points) - 1)]
    split = [x.strip(", ") for x in split]

    return split


def build_app_context(app_entity: AppEntity, query_entity: str) -> AppContext:
    entity_class = entity_to_class(app_entity)
    entity_name = app_entity["entity"]
    app_context = {entity_name: []}

    # Parse the entity
    query_entity = query_entity[query_entity.find("(") + 1 : -1]
    query_entity = split_command_into_slots(query_entity)
    kwargs = {}
    for arg in query_entity:
        assert "=" in arg
        key = arg[: arg.find("=")].strip()
        value = arg[arg.find("=") + 1 :].strip()
        kwargs[key] = value
    app_context[entity_name].append(entity_class(**kwargs))
    return app_context
