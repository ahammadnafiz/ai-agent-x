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
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.chroma import ChromaDb
from sentence_transformers import SentenceTransformer
from typing import List, Union
from agno.vectordb.lancedb import LanceDb, SearchType

# Create a custom wrapper class for SentenceTransformer
class SentenceTransformerEmbedder:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        # Add the dimensions property
        self.dimensions = self.model.get_sentence_embedding_dimension()
    
    def get_embedding(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for text using SentenceTransformer.
        
        Args:
            text: Single string or list of strings to embed
            
        Returns:
            Single embedding vector or list of embedding vectors
        """
        embeddings = self.model.encode(text)
        return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
    
    def get_embedding_and_usage(self, text: Union[str, List[str]]) -> tuple:
        """Generate embeddings for text and return with usage information.
        
        Args:
            text: Single string or list of strings to embed
            
        Returns:
            Tuple of (embedding, usage_info)
            For SentenceTransformer, usage is None since there's no token usage to track
        """
        embedding = self.get_embedding(text)
        # SentenceTransformer doesn't have usage tracking like API-based models
        # So we return None for the usage
        return embedding, None

# create a recipe expert agent
agent = Agent(
    model = Groq(
        id = 'meta-llama/llama-4-scout-17b-16e-instruct',
    ),
    instructions=dedent("""\
        You are a passionate and knowledgeable Thai cuisine expert! 🧑‍🍳
        Think of yourself as a combination of a warm, encouraging cooking instructor,
        a Thai food historian, and a cultural ambassador.

        Follow these steps when answering questions:
        1. First, search the knowledge base for authentic Thai recipes and cooking information
        2. If the information in the knowledge base is incomplete OR if the user asks a question better suited for the web, search the web to fill in gaps
        3. If you find the information in the knowledge base, no need to search the web
        4. Always prioritize knowledge base information over web results for authenticity
        5. If needed, supplement with web searches for:
            - Modern adaptations or ingredient substitutions
            - Cultural context and historical background
            - Additional cooking tips and troubleshooting

        Communication style:
        1. Start each response with a relevant cooking emoji
        2. Structure your responses clearly:
            - Brief introduction or context
            - Main content (recipe, explanation, or history)
            - Pro tips or cultural insights
            - Encouraging conclusion
        3. For recipes, include:
            - List of ingredients with possible substitutions
            - Clear, numbered cooking steps
            - Tips for success and common pitfalls
        4. Use friendly, encouraging language

        Special features:
        - Explain unfamiliar Thai ingredients and suggest alternatives
        - Share relevant cultural context and traditions
        - Provide tips for adapting recipes to different dietary needs
        - Include serving suggestions and accompaniments

        End each response with an uplifting sign-off like:
        - 'Happy cooking! ขอให้อร่อย (Enjoy your meal)!'
        - 'May your Thai cooking adventure bring joy!'
        - 'Enjoy your homemade Thai feast!'

        Remember:
        - Always verify recipe authenticity with the knowledge base
        - Clearly indicate when information comes from web sources
        - Be encouraging and supportive of home cooks at all skill levels\
    """),
    knowledge=PDFUrlKnowledgeBase(
        urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
        vector_db=LanceDb(
            uri = "tmp/recipes",
            table_name='recipe_knowledge',
            search_type=SearchType.hybrid,
            embedder=SentenceTransformerEmbedder()
        ),
    ),
    tools = [
        DuckDuckGoTools(),
    ],
    show_tool_calls=True,
    markdown=True,
    add_references=True,
)

# if agent.knowledge is not None:
#     agent.knowledge.load()


# agent.print_response(
#     "How do I make chicken and galangal in coconut milk soup", stream=True
# )

agent.print_response("What ingredients do I need for Pad Thai?", stream=True)