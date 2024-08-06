import functools
import inspect
import time
import httpx
import json

from typing import Any, Union, Mapping, Callable, get_origin, get_args

from openai              import OpenAI
from openai._types       import NOT_GIVEN, Timeout, NotGiven
from openai._base_client import DEFAULT_MAX_RETRIES
from openai.types.chat.chat_completion import ChatCompletion

from loguru import logger as log


class Agent(OpenAI):

    _GHOSTED_FUNC_PREFIX = "unitialized_"

    _tools_list: list[dict[str, Any]] = []

    _names_to_functions: dict[str, functools.partial] = {}
    
    _name: str | None = None

    _role: str | None = None
    
    _goal: str | None = None
    
    _backstory: str | None = None

    _context: list[dict[str, str]] = []

    def __init__(
        self,
        *,
        name: str | None = None,
        role: str,
        goal: str,
        backstory: str,
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

        self._name      = name
        self._role      = role
        self._goal      = goal
        self._backstory = backstory

        self.build_context()
        
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


    def build_context(self):
        """
        This function builds the context for the agent based on the role, goal, and backstory provided during initialization.
        """

        log.debug(f"Building context for agent {self._name}...")
        log.debug(f"\tRole: {self._role}")
        log.debug(f"\tGoal: {self._goal}")
        log.debug(f"\tBackstory: {self._backstory}")

        self._context.append({
            "role": "system",
            "content": "Your name is : " + self._name
        })
        self._context.append({
            "role": "system",
            "content": "Your role is : " + self._role
        })
        self._context.append({
            "role": "system",
            "content": "Your goal is : " + self._goal
        })
        self._context.append({
            "role": "system",
            "content": "Your backstory is : " + self._backstory
        })
        
        log.debug(f"context : {self._context}")


    def _simple_message(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int | NotGiven = NOT_GIVEN,
        stream: bool = False,
        temperature = 0.8,
    ) -> ChatCompletion:
        """
        Sends a message to the chat API without any tool calls.
        
        Args:
            model (str): The model to use for the completion.
            messages (list[dict[str, str]]): A list of messages to send to the chat API.
            max_tokens (int): The maximum number of tokens to generate.
            stream (bool): Whether to stream the response or wait for the completion.
            temperature (float): The sampling temperature to use for the completion.

        Returns:
            ChatCompletion: The completion response from the chat API.
        """

        return self.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
        )


    def _tool_message(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int | NotGiven = NOT_GIVEN,
        stream: bool = False,
        tool_choice=NOT_GIVEN,
        temperature=0.8
    ) -> ChatCompletion:
        """
        Sends a message to the chat API with a tool call.

        Args:
            model (str): The model to use for the completion.
            messages (list[dict[str, str]]): A list of messages to send to the chat API.
            max_tokens (int): The maximum number of tokens to generate.
            stream (bool): Whether to stream the response or wait for the completion.
            tool_choice (str): The tool choice to use for the completion.
            temperature (float): The sampling temperature to use for the completion.

        Returns:
            ChatCompletion: The completion response from the chat API.
        """

        self.add_tool(
            func=self._simple_message,
            func_description="This function will respond to the user question.",
        )

        output = self.chat.completions.create(
            model = model,
            messages = messages,
            tools = NOT_GIVEN if tool_choice == NOT_GIVEN else self._tools_list,
            tool_choice = "any",
            temperature = temperature,
        )

        messages.append(output.choices[0].message)

        log.debug(output.choices[0].message)

        tool_call = output.choices[0].message.tool_calls[0]
        function_name = tool_call.function.name
        function_params = json.loads(tool_call.function.arguments)

        log.debug("\nfunction_name: " + str(function_name) + "\nfunction_params: " + str(function_params))

        if function_name.startswith(self._GHOSTED_FUNC_PREFIX):

            messages.append({
                "role":"system",
                "content":"The tool : " + function_name + " is not available.",
            })

        elif function_name == self._simple_message.__name__:

            return self._simple_message(
                model = model,
                messages = messages,
                max_tokens = max_tokens,
                temperature = temperature
            )

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
            max_tokens=max_tokens,
            stream=stream
        )

        return output


    def send_message(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int | NotGiven = NOT_GIVEN,
        stream: bool = False,
        tool_choice=NOT_GIVEN,
        temperature=0.8,
    ) -> tuple[Union[str, Any], ChatCompletion]:

        start: float = time.time()
        
        output: ChatCompletion

        if tool_choice == NOT_GIVEN:

            output = self._simple_message(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                stream=stream,
                temperature=temperature
            )

        else:

            output = self._tool_message(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                stream=stream,
                tool_choice=tool_choice,
                temperature=temperature
            )

        messages.append(output.choices[0].message)

        end = time.time()

        log.debug(f"Time taken: {end - start} seconds")

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
        """
        Returns a new partial object which when called will behave like partial called with the positional arguments args and keyword arguments keywords.
        
        Args:
            partial (functools.partial): The partial object to be added as a tool.
        
        Returns:
            bool: True if the tool is successfully added, False if the tool already exists.
        """

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
