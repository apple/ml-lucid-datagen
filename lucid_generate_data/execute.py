#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import inspect
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List

import yaml

from lucid_generate_data.stage import Stage, StageExecutionException, stage_factory


@dataclass
class StageNode:
    stage_object: Stage
    dependency: List[str]


def load_config(config_path: str) -> Dict[str, StageNode]:
    with open(config_path, "r") as rf:
        data = yaml.safe_load(rf)

    stages = OrderedDict()
    for stage in data["pipeline"]:
        stage_name = stage["stage"]
        stage_object = stage_factory(stage_name)
        dependency = stage.get("dependency", [])
        stages[stage_name] = StageNode(stage_object=stage_object, dependency=dependency)

    return stages


def execute(stages: Dict[str, StageNode], trace: Dict[str, Any]) -> None:
    for stage_name, stage_node in stages.items():
        stage_object = stage_node.stage_object
        arg_signatures = inspect.signature(stage_object)

        stage_inputs = {}
        for arg in arg_signatures.parameters:
            if arg == "self":
                continue
            if arg_signatures.parameters[arg].default == inspect.Parameter.empty:
                # Assign value for non-optional args
                if arg not in trace:
                    raise StageExecutionException(f"Value of {arg} is missing in {stage_name}")
                stage_inputs[arg] = trace[arg]
            elif arg in trace:
                # Assign value for optional args with custom input
                stage_inputs[arg] = trace[arg]

        stage_outputs = stage_object(**stage_inputs)
        trace.update(stage_outputs)
