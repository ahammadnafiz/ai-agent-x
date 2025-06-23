from autogen_agentchat.agents import CodeExecutorAgent, AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_core import CancellationToken
from autogen_core.models import UserMessage
from autogen_agentchat.base import TaskResult
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

async def main():
    # Create a local output directory for Docker to mount
    output_dir = "/media/nafiz/NewVolume/ai-agent-x/AutoGen Agents/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set proper permissions for the output directory
    os.chmod(output_dir, 0o777)
    
    # Use a data science Docker image with pre-installed libraries
    docker = DockerCommandLineCodeExecutor(
        work_dir=output_dir,
    )
    
    await docker.start()
    
    model = OpenAIChatCompletionClient(
        model="gemini-2.0-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    code_developer_agent = AssistantAgent(
        name="CodeDeveloperAgent",
        model_client=model,
        system_message="""You are a CodeDeveloperAgent working in a collaborative team with a CodeExecutorAgent. Your role is to analyze programming tasks and generate code solutions that will be executed by your partner agent.

WORKFLOW COLLABORATION:
- You receive programming tasks and create code solutions
- The CodeExecutorAgent will execute your code in a Docker container
- You review execution results and iterate if needed
- Continue the collaboration until the task is successfully completed

CODE GENERATION REQUIREMENTS:
- Always wrap code in proper markdown code blocks with language specification: ```python, ```javascript, etc.
- Write complete, executable code snippets that work independently
- Include necessary imports and dependencies at the top
- Add clear comments explaining the logic and approach
- Handle potential errors with try-catch blocks where appropriate

PROGRAMMING EXPERTISE:
- Python: Focus on clean, pythonic code with proper error handling
- Web Development: HTML, CSS, JavaScript, React, Node.js
- Data Science: pandas, numpy, matplotlib, scikit-learn
- System Programming: File operations, API calls, database interactions
- Algorithms: Sorting, searching, data structures, optimization

RESPONSE STRUCTURE:
1. Brief analysis of the task requirements
2. Explanation of your approach and solution strategy
3. Complete code implementation with comments
4. Expected output or behavior description

QUALITY STANDARDS:
- Production-ready code with proper variable naming
- Efficient algorithms and clean logic flow
- Include input validation where necessary
- Follow language-specific best practices and conventions

TERMINATION PROTOCOL:
- Wait for CodeExecutorAgent to execute your code
- Review execution results and output
- If successful and task is complete, say "TERMINATE"
- If issues arise, debug and provide improved code
- Never say "TERMINATE" until code execution is verified successful

Remember: Your code will run in a Docker environment with data science libraries pre-installed (pandas, numpy, matplotlib, seaborn, scikit-learn). The working directory is mounted to a local 'output' folder, so any files you create will appear there. When saving files (like plots), save them to the current working directory with simple filenames like 'plot.png' or 'results.csv'. Always use absolute paths when saving files to ensure they are written to the correct location.""")
    
    code_executor_agent = CodeExecutorAgent(
        name="CodeExecutorAgent",
        code_executor=docker,
    )
    
    team = RoundRobinGroupChat(
        participants=[code_developer_agent, code_executor_agent],
        termination_condition=TextMentionTermination(
            text="TERMINATE",   
        ),
        max_turns=10
    )
    
    task = '''
    
    Toss a coin 100 times and record the results.
    The results should be stored in a CSV file with two columns: "Toss" and "Result".
    Each row should represent a single toss, with "Heads" or "Tails" as the result.
    The file should be saved in the current working directory as "coin_tosses.csv".
    Ensure the code handles any potential errors and includes comments explaining the logic.
    after that, plot the results using matplotlib.
    The plot should show the number of heads and tails in a bar chart.
    '''
        
    async for message in team.run_stream(task=task):
        if isinstance(message, TaskResult):
            print(f"Stored result: {message.stop_reason}")
        else:
            print(f"Message: {message.source} - {message.content}")
        
    await docker.stop()
    
if __name__ == "__main__":
    asyncio.run(main())