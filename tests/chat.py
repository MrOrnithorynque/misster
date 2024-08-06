from ..src.agent import Agent
from loguru import logger as log


import os


def main():

    misster = Agent(
        name="Misster",
        role="assistant",
        goal="chat with the user",
        backstory="I am a helpful assistant that can perform various tasks.",
        base_url="https://api.mistral.ai/v1",
        api_key=os.getenv("MISTRAL_AI_API_KEY")
    )

    misster._role = "You are a helpful assistant that can perform various tasks."
    
    user_input: str = ""
    messages: list[dict[str, str]] = []
    
    while user_input != "exit":

        user_input = input("You: ")
        
        if user_input == "quit":
            log.info("Exiting...")
            break

        messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        response, output = misster.send_message(
            model="mistral-small-latest",
            messages=messages,
            max_tokens=50,
            temperature=1.3
        )
        
        messages.append(
            {
                "role": "assistant",
                "content": response.__str__()
            }
        )

        print(f"Assistant: {response}")


if __name__ == '__main__':
    main()