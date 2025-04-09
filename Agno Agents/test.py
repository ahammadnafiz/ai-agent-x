import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
groq_api_key = os.environ.get('GROQ_API_KEY')
# Set the environment variables
os.environ['GROQ_API_KEY'] = groq_api_key

from agno.agent import Agent, RunResponse
from agno.models.groq import Groq


# Initialize the Groq model
agent = Agent(
    model = Groq(
        id='llama-3.3-70b-versatile',
        max_tokens=4096,
        temperature=0.7,
    ),
    markdown=True,
)

agent.print_response("Share a 2 sentence horror story.")

