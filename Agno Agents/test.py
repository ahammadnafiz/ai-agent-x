# import os
# from dotenv import load_dotenv

# # Load environment variables from .env file
# load_dotenv()
# groq_api_key = os.environ.get('GROQ_API_KEY')
# # Set the environment variables
# os.environ['GROQ_API_KEY'] = groq_api_key

# from agno.agent import Agent, RunResponse
# from agno.models.groq import Groq


# # Initialize the Groq model
# agent = Agent(
#     model = Groq(
#         id='llama-3.3-70b-versatile',
#         max_tokens=4096,
#         temperature=0.7,
#     ),
#     markdown=True,
# )

# agent.print_response("Share a 2 sentence horror story.")


from agno.agent import AgentKnowledge
from agno.vectordb.pgvector import PgVector
from sentence_transformers import SentenceTransformer
from typing import List, Union

# Create a custom wrapper class for SentenceTransformer
class SentenceTransformerEmbedder:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
    
    def get_embedding(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for text using SentenceTransformer.
        
        Args:
            text: Single string or list of strings to embed
            
        Returns:
            Single embedding vector or list of embedding vectors
        """
        embeddings = self.model.encode(text)
        return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings

# Now use this wrapper in your AgentKnowledge setup
knowledge_base = AgentKnowledge(
    vector_db=PgVector(
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        table_name="huggingface_embeddings",
        embedder=SentenceTransformerEmbedder(),  # Use our custom wrapper
    ),
    num_documents=2,
)