"""
Football Scouting Agent
LangChain ReAct agent powered by Groq (Llama 3.3).
"""

import logging
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from src.agent.prompts import SYSTEM_PROMPT
from src.agent.tools_wrapper import ALL_TOOLS

load_dotenv()
logger = logging.getLogger(__name__)


class ScoutingAgent:
    """AI-powered football scouting assistant."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
    ):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found! Add it to your .env file."
            )

        self.llm = ChatGroq(
            model=model,
            temperature=temperature,
            api_key=api_key,
        )

        self.agent = create_react_agent(
            model=self.llm,
            tools=ALL_TOOLS,
            prompt=SYSTEM_PROMPT,
        )

        self.chat_history = []
        logger.info(f"Scouting Agent initialized with {model}")

    def chat(self, user_message: str) -> str:
        """Send a message and get a response.

        Args:
            user_message: The user's question or request

        Returns:
            Agent's response as string
        """
        self.chat_history.append(HumanMessage(content=user_message))

        response = self.agent.invoke({
            "messages": self.chat_history,
        })

        # Get the last AI message
        ai_messages = [
            msg for msg in response["messages"]
            if hasattr(msg, "content") and msg.type == "ai" and msg.content
        ]

        if ai_messages:
            reply = ai_messages[-1].content
        else:
            reply = "I couldn't process that request. Please try again."

        self.chat_history = response["messages"]
        return reply

    def reset(self):
        """Reset conversation history."""
        self.chat_history = []
        logger.info("Conversation history reset.")


def run_interactive():
    """Run the agent in interactive terminal mode."""
    print("=" * 60)
    print("  Football Scouting Agent")
    print("  Powered by Llama 3.3 via Groq + StatsBomb Data")
    print("=" * 60)
    print("\nType your scouting queries. Type 'quit' to exit.\n")

    agent = ScoutingAgent()

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break

            if user_input.lower() == "reset":
                agent.reset()
                print("Conversation reset.")
                continue

            print("\nAgent: ", end="", flush=True)
            response = agent.chat(user_input)
            print(response)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Try again or type 'reset' to start fresh.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_interactive()
