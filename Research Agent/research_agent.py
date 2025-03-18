import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from crewai import LLM

# Load environment variables
load_dotenv()

# Get API keys
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["SERPER_API_KEY"] = SERPER_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize the search tool
search_tool = SerperDevTool()

def create_research_agent():
    """
    Create a research agent using Groq's LLM instead of OpenAI
    
    Returns:
        Agent: A CrewAI agent configured for research
    """
    # Initialize Groq LLM
    llm = LLM(
    model="groq/llama-3.2-90b-text-preview",
    temperature=0.7
)
    
    # Create and return the agent
    return Agent(
        role="Research Specialist",
        goal="Conduct thorough research on given topics and provide comprehensive, accurate information",
        backstory="""You are an experienced research specialist with expertise in finding, analyzing, 
        and synthesizing information from various sources. You're known for your ability to distill 
        complex topics into clear, accurate summaries backed by evidence.""",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
    )

def create_research_task(agent, topic):
    """
    Create a research task for the agent to perform
    
    Args:
        agent (Agent): The research agent
        topic (str): The research topic
        
    Returns:
        Task: A CrewAI task for research
    """
    return Task(
        description=f"""Research the following topic thoroughly: {topic}
        
        Your research should:
        1. Gather information from multiple reliable sources
        2. Analyze the latest developments and key perspectives
        3. Identify important facts, statistics, and expert opinions
        4. Consider historical context and future implications
        5. Organize findings in a logical structure
        
        When presenting your findings:
        - Begin with a clear, direct summary of the topic
        - Provide context and background information
        - Highlight key facts, trends, and insights
        - Use bullet points for important lists when appropriate
        - Include mentions of relevant research or expert consensus
        - Conclude with implications or future considerations
        """,
        agent=agent,
        expected_output="""A detailed summary of the research findings, including:
        - Executive summary of the topic
        - Key findings and insights with supporting evidence
        - Analysis of different perspectives or approaches
        - Relevant data, statistics, or expert opinions
        - Implications and recommendations based on the research
        """
    )

def run_research(topic):
    """
    Run a research task on the given topic
    
    Args:
        topic (str): The research topic
        
    Returns:
        str: The research results
    """
    agent = create_research_agent()
    task = create_research_task(agent, topic)
    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()
    return result

if __name__ == "__main__":
    print("Welcome to the Research Agent powered by Groq!")
    topic = input("Enter the research topic: ")
    result = run_research(topic)
    print("\nResearch Result:")
    print(result)