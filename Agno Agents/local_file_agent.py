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
from file_tools import (
    list_directory_contents,
    read_file_content,
    write_file_content,
    search_files,
    file_operations
)

# Create an agent with local file operation capabilities
agent = Agent(
    model=Groq(
        id='meta-llama/llama-4-scout-17b-16e-instruct',
    ),
    instructions=dedent("""\
        You are a helpful file management assistant that can help users work with local files.
        You can list directories, read and write files, search for content, and perform file operations.
        Always confirm operations before making changes to the filesystem, especially deletions.
        When displaying file contents, format code appropriately and handle binary files safely.
    """),
    tools=[
        list_directory_contents,
        read_file_content,
        write_file_content,
        search_files,
        file_operations
    ],
    show_tool_calls=True,
    markdown=True,
)

# List files in the current directory
# agent.print_response("Show me the files in my current directory")

# Read a specific file
# agent.print_response("Show me the contents of test.py")

# # Search for Python files containing a specific function
agent.print_response("Find all Python files with 'from textwrap import dedent' in them")

# # Create a new file
# agent.print_response("Create a new file called notes.txt with 'Meeting notes for project X' as the content")

# # Move a file to another location
# agent.print_response("Move report.pdf to the archive folder")