from ..src.agent import Agent
from loguru import logger as log

import os


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

    misster = Agent(
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