from dotenv import load_dotenv
import os
from typing import Iterator

# Load environment variables from .env file
load_dotenv()
gemini_api_key = os.environ.get('GOOGLE_API_KEY')
# Set the environment variables
os.environ['GOOGLE_API_KEY'] = gemini_api_key

from textwrap import dedent

from agno.agent import Agent, RunResponse, RunResponseEvent
from agno.models.google import Gemini
from agno.tools.yfinance import YFinanceTools
from agno.utils.pprint import pprint_run_response
from rich.pretty import pprint

agent = Agent(
    model=Gemini(id="gemini-2.0-flash-001"),
    tools=[YFinanceTools(stock_price=True)],
    markdown=True,
    show_tool_calls=True,
)

agent.print_response(
    "What is the stock price of NVDA", stream=True
)

# Print metrics per message
if agent.run_response.messages:
    for message in agent.run_response.messages:
        if message.role == "assistant":
            if message.content:
                print(f"Message: {message.content}")
            elif message.tool_calls:
                print(f"Tool calls: {message.tool_calls}")
            print("---" * 5, "Metrics", "---" * 5)
            pprint(message.metrics)
            print("---" * 20)

# Print the aggregated metrics for the whole run
print("---" * 5, "Collected Metrics", "---" * 5)
pprint(agent.run_response.metrics)
# Print the aggregated metrics for the whole session
print("---" * 5, "Session Metrics", "---" * 5)
pprint(agent.session_metrics)


