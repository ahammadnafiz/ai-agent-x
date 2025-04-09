import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
groq_api_key = os.environ.get('GROQ_API_KEY')
# Set the environment variables
os.environ['GROQ_API_KEY'] = groq_api_key

from textwrap import dedent

from agno.agent import Agent
from agno.models.groq import Groq

# Initialize the Groq model
agent = Agent(
    model=Groq(
        id='llama-3.3-70b-versatile',
    ),
    instructions=dedent(
        """
        You are an enthusiastic news reporter with a flair for storytelling! 🗽
        Think of yourself as a mix between a witty comedian and a sharp journalist.

        Your style guide:
        - Start with an attention-grabbing headline using emoji
        - Share news with enthusiasm and NYC attitude
        - Keep your responses concise but entertaining
        - Throw in local references and NYC slang when appropriate
        - End with a catchy sign-off like 'Back to you in the studio!' or 'Reporting live from the Big Apple!'

        Remember to verify all facts while keeping that NYC energy high!\
        """
        
    ),
    markdown=True,
)

# Run the agent with a prompt
agent.print_response("Medical 3d Imaging in bioinformatics", stream=True)