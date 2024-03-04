#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from jinja2 import Environment
from textwrap import dedent


def prompt_template_nlg() -> Environment:
    environment = Environment()
    return environment.from_string(
        dedent(
            """
	    You are a smart AI assistant who is responsible for creating a natural language response back to the user. A response is required after the `say' call, which must be run as `say(x, ...)` where `x` is a relevant entity to mention. 

        Example conversations with natural language responses:
        {{examples}}

	    Now it's your turn.
	    {%- if incomplete %} You can only use the variables:
	    {%- for inc in incomplete %} {{ inc }}
	    {%- if loop.revindex0==1 %} and {%- elif not loop.last %}, {%- endif -%}
	    {% endfor -%}
	    {% endif %}

        {%- if result %}
        The query has returned the following results, which should inform your response to the user:
	    {{result}}
	    {%- endif %}

        {{conversation_text}}

	    {{next_index}}{{lucid_prompt}}
	    """
        ).lstrip()
    )
