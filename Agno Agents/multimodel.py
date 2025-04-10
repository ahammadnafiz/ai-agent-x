import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

from agno.agent import Agent
from agno.models.google import Gemini
from agno.media import Image
from agno.tools.duckduckgo import DuckDuckGoTools

# Image Agent
agent = Agent(
    model = Gemini(
        id = "gemini-2.0-flash",
        api_key = api_key,
    ),
    tools = [DuckDuckGoTools()],
    markdown=True,
)

agent.print_response(
    "Tell me about this image and give me the latest news about it.",
    images=[
        Image(
            url="https://upload.wikimedia.org/wikipedia/commons/0/0c/GoldenGateBridge-001.jpg",
        )
    ],
    stream=True,
)

