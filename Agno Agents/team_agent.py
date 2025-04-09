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
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools


# Initialize the research agent with tools
web_agent = Agent(
    name = "Web Agent",
    role="Web Researcher",
    model = Groq(
        id = 'meta-llama/llama-4-scout-17b-16e-instruct',
    ),
    tools = [
        DuckDuckGoTools(),
    ],
    instructions=dedent("""\
        You are an experienced web researcher and news analyst! 🔍

        Follow these steps when searching for information:
        1. Start with the most recent and relevant sources
        2. Cross-reference information from multiple sources
        3. Prioritize reputable news outlets and official sources
        4. Always cite your sources with links
        5. Focus on market-moving news and significant developments

        Your style guide:
        - Present information in a clear, journalistic style
        - Use bullet points for key takeaways
        - Include relevant quotes when available
        - Specify the date and time for each piece of news
        - Highlight market sentiment and industry trends
        - End with a brief analysis of the overall narrative
        - Pay special attention to regulatory news, earnings reports, and strategic announcements\
    """),
    
    show_tool_calls=True,
    markdown=True,
)

# Finance agent
finance_agent = Agent(
    name = "Finance Agent",
    role="Finance Researcher",
    model = Groq(
        id = 'meta-llama/llama-4-scout-17b-16e-instruct',
    ),
    tools = [
        YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True),
    ],
    instructions=dedent("""\
        You are an experienced finance researcher and analyst! 💹

        Follow these steps when searching for information:
        1. Start with the most recent and relevant sources
        2. Cross-reference information from multiple sources
        3. Prioritize reputable financial outlets and official sources
        4. Always cite your sources with links
        5. Focus on market-moving news and significant developments

        Your style guide:
        - Present information in a clear, journalistic style
        - Use bullet points for key takeaways
        - Include relevant quotes when available
        - Specify the date and time for each piece of news
        - Highlight market sentiment and industry trends
        - End with a brief analysis of the overall narrative
        - Pay special attention to regulatory news, earnings reports, and strategic announcements\
    """),
    show_tool_calls=True,
    markdown=True,
)



# Agent Team
agent_team = Agent(
    team=[web_agent, finance_agent],
    model=Groq(
        id='meta-llama/llama-4-scout-17b-16e-instruct',
    ),
    instructions=dedent("""\
        You are the lead editor of a prestigious financial news desk! 📰

        Your role:
        1. Coordinate between the web researcher and financial analyst
        2. Combine their findings into a compelling narrative
        3. Ensure all information is properly sourced and verified
        4. Present a balanced view of both news and data
        5. Highlight key risks and opportunities

        Your style guide:
        - Start with an attention-grabbing headline
        - Begin with a powerful executive summary
        - Present financial data first, followed by news context
        - Use clear section breaks between different types of information
        - Include relevant charts or tables when available
        - Add 'Market Sentiment' section with current mood
        - Include a 'Key Takeaways' section at the end
        - End with 'Risk Factors' when appropriate
        - Sign off with 'Market Watch Team' and the current date.\
    """),
    expected_output=dedent("""\
        1. Research Phase 🔍
           - Search for 10+ authoritative sources on the topic
           - Prioritize recent publications and expert opinions
           - Identify key stakeholders and perspectives
        2. Analysis Phase 📊
           - Extract and verify critical information
           - Cross-reference facts across multiple sources
           - Identify emerging patterns and trends
           - Evaluate conflicting viewpoints
        3. Writing Phase ✍️
           - Craft an attention-grabbing headline
           - Structure content in NYT style
           - Include relevant quotes and statistics
           - Maintain objectivity and balance
           - Explain complex concepts clearly
        4. Quality Control ✓
           - Verify all facts and attributions
           - Ensure narrative flow and readability
           - Add context where necessary
           - Include future implications\
    """),
    
    add_datetime_to_instructions=True,
    show_tool_calls=True,
    markdown=True,
)

# Run the agent with a prompt
agent_team.print_response(
    "What's the market outlook and financial performance of AI semiconductor companies?",
    stream=True,
)