import functools
import inspect
import time
import httpx
import json
import os

from typing import Any, Union, Mapping, Callable, get_origin, get_args

from openai              import OpenAI
from openai._types       import NOT_GIVEN, Timeout, NotGiven
from openai._base_client import DEFAULT_MAX_RETRIES
from openai.types.chat.chat_completion import ChatCompletion

from loguru import logger as log


class Misster(OpenAI):

    _GHOSTED_FUNC_PREFIX = "unitialized_"

    _tools_list: list[dict[str, Any]] = []

    _names_to_functions: dict[str, functools.partial] = {}

    _role: str = ""
    """
    What is his role as LLM? Describe it here.
    """

    _context: list[dict[str, str]] = []

    def __init__(
        self,
        *,
        api_key: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        base_url: str | httpx.URL | None = None,
        timeout: Union[float, Timeout, None, NotGiven] = NOT_GIVEN,
        max_retries: int = DEFAULT_MAX_RETRIES,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        # Configure a custom httpx client.
        # We provide a `DefaultHttpxClient` class that you can pass to retain the default values we use for `limits`, `timeout` & `follow_redirects`.
        # See the [httpx documentation](https://www.python-httpx.org/api/#client) for more details.
        http_client: httpx.Client | None = None,
        # Enable or disable schema validation for data returned by the API.
        # When enabled an error APIResponseValidationError is raised
        # if the API responds with invalid data for the expected schema.
        #
        # This parameter may be removed or changed in the future.
        # If you rely on this feature, please open a GitHub issue
        # outlining your use-case to help us decide if it should be
        # part of our public interface in the future.
        _strict_response_validation: bool = False,
    ) -> None:

        super().__init__(
            api_key=api_key,
            organization=organization,
            project=project,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            _strict_response_validation=_strict_response_validation,
        )

        return None


    def send_message(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int | NotGiven = NOT_GIVEN,
        stream: bool = False,
        tool_choice=NOT_GIVEN,
        temperature=0.8,
        save_context: bool = False,
    ) -> tuple[Union[str, Any], ChatCompletion]:

        start = time.time()
        
        output = self.chat.completions.create(
            model=model,
            messages=messages,
            tools=NOT_GIVEN if tool_choice == NOT_GIVEN else self._tools_list,
            tool_choice="any",
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        messages.append(output.choices[0].message)
        
        # log.info("Response: ")
        # log.info(output.choices[0].message)
        
        if output.choices[0].message.tool_calls:

            log.debug(output.choices[0].message)

            log.info("Tool call detected in response.")

            tool_call = output.choices[0].message.tool_calls[0]
            function_name = tool_call.function.name
            function_params = json.loads(tool_call.function.arguments)
            log.info("\nfunction_name: " + str(function_name) + "\nfunction_params: " + str(function_params))

            if function_name.startswith(self._GHOSTED_FUNC_PREFIX):

                messages.append({
                    "role":"system",
                    "content":"This tool is not available.",
                })

            else:

                function_result = self._names_to_functions[function_name](**function_params)

                messages.append({
                    "role":"tool",
                    "name":function_name,
                    "content":str(function_result),
                    "tool_call_id":tool_call.id
                })

                log.debug("Tool call result :")
                log.debug(function_result)

            output = self.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            messages.append(output.choices[0].message)
            
            if not save_context:
                messages.pop()
                messages.pop()

        end = time.time()
        
        if not save_context:
            messages.pop()
        
        log.info(f"Time taken: {end - start} seconds")

        return (None, output) if stream else (output.choices[0].message.content, output)


    def _get_param_openai_type(self, param: inspect.Parameter) -> str | dict[str, Any]:
        """
        Determines the OpenAI-compatible JSON type for a given function parameter.

        This function maps Python types to JSON-compatible types that are used by OpenAI. 
        It also handles complex types such as arrays (list, tuple) by specifying the type of 
        items within the array.

        Args:
            param (inspect.Parameter): The function parameter for which to determine the JSON type.

        Returns:
            str or dict: A string representing the JSON type or a dictionary for complex types 
                        like arrays. For arrays, the dictionary includes the type and item 
                        specifications.

        Supported Types:
            - int: "number"
            - float: "number"
            - str: "string"
            - bool: "boolean"
            - None: "null"
            - list: "array" (with item type description)
            - tuple: "array" (with item type description)
            - dict: "object"

        If the parameter type is not recognized, a warning is logged, and "object" is returned as the default type.
        """

        type_mapping = {
            int: "number",
            float: "number",
            str: "string",
            bool: "boolean",
            type(None): "null",
            list: "array",
            tuple: "array",
            dict: "object",
        }

        param_type = param.annotation

        if get_origin(param_type) in [list, tuple]:

            item_type = get_args(param_type)[0]
            item_json_type = type_mapping.get(item_type, "object")

            return {
                "type": "array",
                "items": {
                    "type": item_json_type,
                    "description": f"Array of {item_json_type}"
                }
            }

        if param_type not in type_mapping:
            log.warning(f"Type {param_type} not supported by automatic type mapping. Defaulting to 'object'.")
            return "object"

        return type_mapping[param_type]


    def add_tool(
        self,
        *,
        func: Callable,
        func_description: str,
        params_description: dict[str, Union[str, tuple[str, str]]] = {},
        reroll_func_result: bool = True
    ) -> bool:
        """
        Adds a tool (function) to the tool manager, including its description and parameter descriptions.

        This function registers a tool (function) with the tool manager, providing a description of 
        the function and descriptions for each parameter. It ensures that only parameters existing 
        in the function signature are added and handles complex types such as arrays by specifying 
        the type of items within the array.

        Args:
            func (Callable): The function to be added as a tool.
            func_description (str): A description of what the function does.
            params_description (dict[str, Union[str, tuple[str, str]]]): A dictionary where keys are parameter names and 
                                                                        values are descriptions of those parameters. If the value 
                                                                        is a tuple, the first element is the parameter description, 
                                                                        and the second element is the description of the array items.

        Returns:
            bool: True if the tool is successfully added, False if the tool already exists.

        Example:
            def add(a: int, b: int) -> int:
                return a + b

            tool_manager.add_tool(
                func=add,
                func_description="Add two numbers",
                params_description={
                    "a": "First number.",
                    "b": "Second number."
                }
            )

        The parameters dictionary is constructed with the appropriate types and descriptions, 
        including handling for array types. The tool is appended to the `_tools_list` attribute.
        """

        sig = inspect.signature(func)
        func_params = sig.parameters

        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }

        for param, desc in params_description.items():

            if param in func_params:

                param_type = self._get_param_openai_type(func_params[param])
                param_details = {
                    "type": param_type if isinstance(param_type, str) else param_type["type"],
                    "description": desc if isinstance(desc, str) else desc[0],
                }

                if isinstance(param_type, dict) and param_type["type"] == "array":
                    param_details["items"] = {
                        "type": param_type["items"]["type"],
                        "description": desc[1] if isinstance(desc, tuple) and len(desc) > 1 else param_type["items"]["description"]
                    }

                parameters["properties"][param] = param_details
                if func_params[param].default == inspect.Parameter.empty:
                    parameters["required"].append(param)

        self._tools_list.append({
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": func_description,
                "parameters": parameters,
            },
        })

        if len(func_params) == len(params_description):
            self._names_to_functions[func.__name__] = functools.partial(func)
            log.info(f"Tool '{func.__name__}' added successfully.")
        else:
            # ghost func.__name__
            self._tools_list[-1]["function"]["name"] = self._GHOSTED_FUNC_PREFIX + func.__name__
            log.debug(f"Function '{func.__name__}' has a different number of parameters than the description. You should add them manually.")

        log.debug(self._tools_list[-1])

        return True


    def tool_partial(self, partial: functools.partial) -> bool:
        
        func_name: str = partial.func.__name__

        if func_name in self._names_to_functions:
            log.warning(f"Function '{func_name}' already exists in the tool manager.")
            return False

        self._names_to_functions[func_name] = partial
        
        for tool in self._tools_list:
            if tool["function"]["name"] == self._GHOSTED_FUNC_PREFIX + func_name:
                tool["function"]["name"] = func_name

        log.info(f"Tool '{func_name}' added successfully.")

        return True


    def remove_tool(
        self,
        *,
        tool_name: str
    ) -> bool:

        for tool in self._tools_list:
            if tool["function"]["name"] == tool_name:

                self._tools_list.remove(tool)
                
                if tool_name in self._names_to_functions:
                    del self._names_to_functions[tool_name]

                return True

        return False


    @property
    def _context(
        self,
    ) -> str:
        return None


### TEST MISSTER


MISTRAL_BASE_URL = "https://api.mistral.ai/v1"

MISTRAL_AI_API_KEY = os.getenv("MISTRAL_AI_API_KEY")

messages = [
    {
        "role":"user",
        "content":"1 +1"
    }
]

def add(a: int, b: int) -> int:

    return a + b


def add_array_to_number(a: list[int], b: int) -> list[int]:

    return [x + b for x in a]


def get_data(id: int, db):

    log.info(f"Getting data from database from id {id} in database {db}")
    return

def respond_to_user() -> str:
    
        return "Assistant said"

def test_misster():

    misster = Misster(
        base_url=MISTRAL_BASE_URL,
        api_key=MISTRAL_AI_API_KEY,
    )
    
    misster._role = "You are a helpful assistant that can perform various tasks."
    
    misster.add_tool(
        func=respond_to_user,
        func_description="Respond to a user message",
    )

    misster.add_tool(
        func=add,
        func_description="Add two numbers",
        params_description={
            "a": "First number.",
            "b": "Second number."
        }
    )

    misster.add_tool(
        func=add_array_to_number,
        func_description="Add a number to each element of an array",
        params_description={
            "a": ["Array of numbers.", "a number"],
            "b": "Number to add to each element."
        }
    )

    misster.add_tool(
        func=get_data,
        func_description="Get data from database",
        params_description={
            "id": "The id of the data to retrieve."
        }
    )

    misster.tool_partial(functools.partial(get_data, db="database"))

    log.info(misster._context)

    response, output = misster.send_message(
        model="mistral-large-latest",
        messages=messages,
        max_tokens=60,
        tool_choice="auto"
    )
    
    log.info("Response:")
    log.info(response)


if __name__ == '__main__':
    test_misster()