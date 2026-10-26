
# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

import asyncio  # Allows us to run asynchronous Python functions.
import os       # Allows us to read environment variables.

from dotenv import load_dotenv
# Loads environment variables stored in the .env file.

from autogen_agentchat.agents import AssistantAgent
# Imports AssistantAgent, which creates our AI assistant.

from autogen_agentchat.ui import Console
# Console displays the agent's response in the terminal.

from autogen_ext.models.openai import OpenAIChatCompletionClient
# Creates a client to communicate with an OpenAI language model.


# ============================================================
# 2. DEFINE THE ASYNCHRONOUS MAIN FUNCTION
# ============================================================

async def main():

    # Load the variables stored in the .env file.
    load_dotenv()

    # Read the OpenAI API key securely from the environment.
    # Never write your actual API key directly in this Python file.
    api_key = os.getenv("OPENAI_API_KEY")

    # Stop execution with a helpful message if the API key is missing.
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is missing. Please add it to your .env file."
        )

    # ========================================================
    # 3. CREATE THE OPENAI MODEL CLIENT
    # ========================================================

    # Connect AutoGen to the OpenAI model.
    # The API key comes from the environment variable above.
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=api_key,
    )

    try:
        # ====================================================
        # 4. CREATE THE HR ASSISTANT AGENT
        # ====================================================

        # Create an AssistantAgent with the exact name and
        # system message required by your assignment.
        agent = AssistantAgent(
            name="hr_assistant",
            model_client=model_client,
            system_message="You are an HR assistant.",
        )

        # ====================================================
        # 5. DEFINE THE TASK FOR THE AGENT
        # ====================================================

        # Explicitly request five numbered lines.
        # This removes ambiguity about what "5 lines" means.
        task = (
            "Write a job description for a Junior Python Developer. "
            "The reply must contain exactly five numbered lines, "
            "with one sentence per line. Do not add a title or "
            "any extra text."
        )

        # ====================================================
        # 6. RUN THE AGENT AND DISPLAY ITS RESPONSE
        # ====================================================

        # run_stream() runs the agent and streams its messages.
        # Console displays those messages in the terminal.
        await Console(agent.run_stream(task=task))

    finally:
        # ====================================================
        # 7. CLOSE THE MODEL CLIENT
        # ====================================================

        # Always attempt to close the client when finished,
        # including when an error occurs inside the try block.
        await model_client.close()


# ============================================================
# 8. START THE PROGRAM
# ============================================================

# Run main() using Python's asynchronous event loop.
# This is the standard entry point for this script.
if __name__ == "__main__":
    asyncio.run(main())
