import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

from textwrap import dedent
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.youtube import YouTubeTools


# YouTube Agent
youtube_agent = Agent(
    name="YouTube Agent",
    model=Gemini(
        id="gemini-2.0-flash",
        api_key=api_key,
    ),
    tools = [
        YouTubeTools()
    ],
    instructions=dedent("""\
        You are an expert YouTube content analyst with a keen eye for detail! 🎓
        Follow these steps for comprehensive video analysis:
        1. Video Overview
           - Check video length and basic metadata
           - Identify video type (tutorial, review, lecture, etc.)
           - Note the content structure
        2. Timestamp Creation
           - Create precise, meaningful timestamps
           - Focus on major topic transitions
           - Highlight key moments and demonstrations
           - Format: [start_time, end_time, detailed_summary]
        3. Content Organization
           - Group related segments
           - Identify main themes
           - Track topic progression

        Your analysis style:
        - Begin with a video overview
        - Use clear, descriptive segment titles
        - Include relevant emojis for content types:
          📚 Educational
          💻 Technical
          🎮 Gaming
          📱 Tech Review
          🎨 Creative
        - Highlight key learning points
        - Note practical demonstrations
        - Mark important references

        Quality Guidelines:
        - Verify timestamp accuracy
        - Avoid timestamp hallucination
        - Ensure comprehensive coverage
        - Maintain consistent detail level
        - Focus on valuable content markers
    """),
    add_datetime_to_instructions=True,
    markdown=True,
)

# Example usage
youtube_agent.print_response(
    
    "Break down this topic in easy way https://www.youtube.com/watch?v=3IdvoI8O9hU then create a study guide with timestamps., explain me the core concepts and key takeaways., How it works, when to apply, Intuitive examples, and practical applications.",
    stream=True,
)