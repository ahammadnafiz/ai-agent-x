import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
groq_api_key = os.environ.get('GROQ_API_KEY')
# Set the environment variables
os.environ['GROQ_API_KEY'] = groq_api_key

from textwrap import dedent
from agno.agent import Agent, RunResponse
from agno.models.groq import Groq
from typing import List
from pydantic import BaseModel, Field

# class MovieScript(BaseModel):
#     setting: str = Field(
#         ...,
#         description="A richly detailed, atmospheric description of the movie's primary location and time period. Include sensory details and mood.",
#     )
#     ending: str = Field(
#         ...,
#         description="The movie's powerful conclusion that ties together all plot threads. Should deliver emotional impact and satisfaction.",
#     )
#     genre: str = Field(
#         ...,
#         description="The film's primary and secondary genres (e.g., 'Sci-fi Thriller', 'Romantic Comedy'). Should align with setting and tone.",
#     )
#     name: str = Field(
#         ...,
#         description="An attention-grabbing, memorable title that captures the essence of the story and appeals to target audience.",
#     )
#     characters: List[str] = Field(
#         ...,
#         description="4-6 main characters with distinctive names and brief role descriptions (e.g., 'Sarah Chen - brilliant quantum physicist with a dark secret').",
#     )
#     storyline: str = Field(
#         ...,
#         description="A compelling three-sentence plot summary: Setup, Conflict, and Stakes. Hook readers with intrigue and emotion.",
#     )
    
# # Agent Json Mode
# json_mode_agent = Agent(
#     model=Groq(
#         id='meta-llama/llama-4-scout-17b-16e-instruct',
#     ),
#     description=dedent("""\
#         You are an acclaimed Hollywood screenwriter known for creating unforgettable blockbusters! 🎬
#         With the combined storytelling prowess of Christopher Nolan, Aaron Sorkin, and Quentin Tarantino,
#         you craft unique stories that captivate audiences worldwide.

#         Your specialty is turning locations into living, breathing characters that drive the narrative.\
#     """),
#     instructions=dedent("""\
#         When crafting movie concepts, follow these principles:

#         1. Settings should be characters:
#            - Make locations come alive with sensory details
#            - Include atmospheric elements that affect the story
#            - Consider the time period's impact on the narrative

#         2. Character Development:
#            - Give each character a unique voice and clear motivation
#            - Create compelling relationships and conflicts
#            - Ensure diverse representation and authentic backgrounds

#         3. Story Structure:
#            - Begin with a hook that grabs attention
#            - Build tension through escalating conflicts
#            - Deliver surprising yet inevitable endings

#         4. Genre Mastery:
#            - Embrace genre conventions while adding fresh twists
#            - Mix genres thoughtfully for unique combinations
#            - Maintain consistent tone throughout

#         Transform every location into an unforgettable cinematic experience!\
#     """),
#     response_model=MovieScript,
# )

# # Structured Output Agent

# structured_output_agent = Agent(
#     model=Groq(
#         id='meta-llama/llama-4-scout-17b-16e-instruct',
#     ),
#     description=dedent("""\
#         You are an acclaimed Hollywood screenwriter known for creating unforgettable blockbusters! 🎬
#         With the combined storytelling prowess of Christopher Nolan, Aaron Sorkin, and Quentin Tarantino,
#         you craft unique stories that captivate audiences worldwide.

#         Your specialty is turning locations into living, breathing characters that drive the narrative.\
#     """),
#     instructions=dedent("""\
#         When crafting movie concepts, follow these principles:

#         1. Settings should be characters:
#            - Make locations come alive with sensory details
#            - Include atmospheric elements that affect the story
#            - Consider the time period's impact on the narrative

#         2. Character Development:
#            - Give each character a unique voice and clear motivation
#            - Create compelling relationships and conflicts
#            - Ensure diverse representation and authentic backgrounds

#         3. Story Structure:
#            - Begin with a hook that grabs attention
#            - Build tension through escalating conflicts
#            - Deliver surprising yet inevitable endings

#         4. Genre Mastery:
#            - Embrace genre conventions while adding fresh twists
#            - Mix genres thoughtfully for unique combinations
#            - Maintain consistent tone throughout

#         Transform every location into an unforgettable cinematic experience!\
#     """),
#     expected_output=MovieScript,
# )

# # json_mode_agent.print_response("Dubai 2050", stream=True)
# structured_output_agent.print_response("Dubai 2050", stream=True)



class User(BaseModel):
    name: str
    age: int
    email: str

agent = Agent(
    model=Groq(
        id='meta-llama/llama-4-scout-17b-16e-instruct',
    ),
    description="You are a helpful assistant that can extract information from a user's profile.",
    # response_model=User,
    expected_output=User,
)

agent.print_response(
    "Please provide your name, age, and email address.",
    stream=True,
)