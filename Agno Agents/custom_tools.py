import json
import re
from datetime import datetime, timedelta
import httpx
from typing import List, Dict, Any, Optional
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


def scan_latest_research_papers(topic: str = "AI", limit: int = 5) -> str:
    """Fetch recent research papers from arXiv on a specific topic.
    
    Args:
        topic (str): Research topic to search for (e.g., "AI", "machine learning", "quantum computing")
        limit (int): Maximum number of papers to return
        
    Returns:
        str: JSON string containing paper details including title, authors, abstract, and URL
    """
    # Convert topic to arXiv search query format
    search_query = topic.replace(" ", "+")
    
    # Set up parameters for arXiv API
    base_url = "http://export.arxiv.org/api/query?"
    params = {
        "search_query": f"all:{search_query}",
        "start": 0,
        "max_results": limit,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    
    # Construct the full URL
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    full_url = base_url + query_string
    
    try:
        # Fetch data from arXiv API
        response = httpx.get(full_url)
        response.raise_for_status()
        
        # Parse XML response
        content = response.text
        
        # Simple regex-based extraction (in production, use a proper XML parser)
        papers = []
        
        # Find entries
        entries = re.findall(r'<entry>(.*?)</entry>', content, re.DOTALL)
        
        for entry in entries[:limit]:
            # Extract paper details
            title_match = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
            title = title_match.group(1).strip() if title_match else "No title available"
            
            authors = []
            author_matches = re.findall(r'<author>.*?<name>(.*?)</name>.*?</author>', entry, re.DOTALL)
            for author in author_matches:
                authors.append(author.strip())
            
            abstract_match = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
            abstract = abstract_match.group(1).strip() if abstract_match else "No abstract available"
            
            url_match = re.search(r'<id>(.*?)</id>', entry)
            url = url_match.group(1).strip() if url_match else "No URL available"
            
            published_match = re.search(r'<published>(.*?)</published>', entry)
            published = published_match.group(1).strip() if published_match else "No date available"
            
            papers.append({
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "url": url,
                "published_date": published,
            })
        
        return json.dumps(papers)
    
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch papers: {str(e)}"})


# Create a Tech News Reporter Agent with a Silicon Valley personality
agent = Agent(
    model=Groq(
        id='meta-llama/llama-4-scout-17b-16e-instruct',
    ),
    instructions=dedent("""\
    You are TechPulse, a cutting-edge tech reporter breaking down the innovation frontier! 🚀
    
    Your personality combines:
    - Silicon Valley insider knowledge with journalistic clarity
    - Technical expertise with approachable explanations
    - Data-driven analysis with engaging storytelling
    
    Content approach:
    - Begin with a bold, emoji-rich headline that captures the tech zeitgeist
    - Transform HN stories, research papers, and Product Hunt discoveries into cohesive narratives
    - Identify connections between trending topics and broader tech ecosystem patterns
    - Highlight practical applications and potential impact of new technologies
    - Surface underrated findings that deserve more attention
    
    Writing style:
    - Use tech industry buzzwords and startup terminology authentically
    - Keep explanations concise yet comprehensive (aim for 2-3 sentences per key point)
    - Mix data points with contextual insights
    - Incorporate occasional tech humor and cultural references
    - Format information with strategic bold text and bullet points for readability
    
    Your response structure:
    1. Eye-catching headline with relevant emoji
    2. Brief trend analysis connecting the data points
    3. Highlighted findings organized by relevance/impact
    4. Brief conclusion with forward-looking perspective
    5. Sign off with a memorable tech-themed phrase like "Compiling future.sh!" or "Deploying insights to production!"
    
    Remember: Your ultimate value is separating signal from noise in the tech ecosystem while maintaining contagious enthusiasm for innovation!
"""),
    tools=[
        scan_latest_research_papers,
    ],
    show_tool_calls=True,
    markdown=True,
)

# Test the agent with a sample query
agent.print_response("What are the latest research papers on AI?", stream=True)