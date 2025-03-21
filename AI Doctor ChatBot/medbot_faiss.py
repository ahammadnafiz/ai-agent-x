"""
MedBot: A medical chatbot system using document retrieval and LLM-based question answering.

This module implements a document retrieval and question answering system specialized for medical
information. It loads PDF documents, processes them, creates embeddings, and stores them in
a FAISS vector database for efficient retrieval. Enhanced with LangChain memory components for
better context retention.
"""

import os
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

# Third-party imports
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# Memory imports
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationSummaryBufferMemory
from langchain.memory.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


class ConfigManager:
    """Manages configuration and environment variables."""
    
    def __init__(self):
        """Initialize configuration manager and load environment variables."""
        # Change to parent directory for correct .env file location
        original_dir = os.getcwd()
        os.chdir('../')
        
        # Load environment variables
        load_dotenv()
        
        # Store API keys
        self.groq_api_key = os.environ.get('GROQ_API_KEY')
        self.redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379')
        
        # Set environment variables for dependent libraries
        os.environ["GROQ_API_KEY"] = self.groq_api_key
        
        # Return to original directory
        os.chdir(original_dir)


class DocumentProcessor:
    """Handles document loading and processing."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 20):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Size of text chunks for splitting documents
            chunk_overlap: Overlap between chunks to maintain context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_pdf_documents(self, directory_path: str) -> List:
        """
        Load PDF documents from a directory.
        
        Args:
            directory_path: Path to directory containing PDF files
            
        Returns:
            List of loaded documents
        """
        loader = DirectoryLoader(
            directory_path,
            glob="*.pdf",
            loader_cls=PyPDFLoader
        )
        return loader.load()
    
    def split_documents(self, documents: List) -> List:
        """
        Split documents into smaller chunks for processing.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        return text_splitter.split_documents(documents)


class EmbeddingManager:
    """Handles document embeddings creation and storage."""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize embedding manager.
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        self.embeddings = None
    
    def load_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Load the embedding model.
        
        Returns:
            HuggingFaceEmbeddings object
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        return self.embeddings


class FAISSManager:
    """Manages FAISS vector database operations."""
    
    def __init__(self, embeddings, index_path: str = "faiss_index"):
        """
        Initialize FAISS manager.
        
        Args:
            embeddings: Embeddings model to use
            index_path: Path to store/load FAISS index
        """
        self.embeddings = embeddings
        self.index_path = index_path
        self.vector_store = None
    
    def create_index(self, documents: List) -> None:
        """
        Create a new FAISS index from documents.
        
        Args:
            documents: List of document chunks to index
        """
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        self._save_index()
    
    def _save_index(self) -> None:
        """Save the FAISS index to disk."""
        if self.vector_store:
            self.vector_store.save_local(self.index_path)
    
    def load_index(self) -> None:
        """Load existing FAISS index from disk."""
        self.vector_store = FAISS.load_local(
            folder_path=self.index_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
    
    def get_retriever(self, search_type: str = "similarity", k: int = 3):
        """
        Create a retriever from the vector store.
        
        Args:
            search_type: Type of search to perform
            k: Number of results to return
            
        Returns:
            Retriever object
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Load or create an index first.")
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )


class MemoryManager:
    """Manages different types of conversation memory."""
    
    def __init__(self, memory_type: str = "buffer", redis_url: Optional[str] = None, 
                 llm = None, max_token_limit: int = 2000):
        """
        Initialize memory manager with specified memory type.
        
        Args:
            memory_type: Type of memory to use ("buffer", "summary", "summary_buffer", "redis")
            redis_url: URL for Redis connection (required for redis memory type)
            llm: Language model for summarization (required for summary memory types)
            max_token_limit: Maximum number of tokens to keep in memory
        """
        self.memory_type = memory_type
        self.redis_url = redis_url
        self.llm = llm
        self.max_token_limit = max_token_limit
        self.memory = None
    
    def create_memory(self, session_id: str = "default") -> Any:
        """
        Create and return the appropriate memory object.
        
        Args:
            session_id: Unique identifier for the chat session
            
        Returns:
            Memory object based on the specified type
        """
        if self.memory_type == "buffer":
            self.memory = ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history",
                output_key="answer"
            )
        
        elif self.memory_type == "summary":
            if not self.llm:
                raise ValueError("LLM is required for summary memory")
                
            self.memory = ConversationSummaryMemory(
                llm=self.llm,
                return_messages=True,
                memory_key="chat_history",
                output_key="answer"
            )
            
        elif self.memory_type == "summary_buffer":
            if not self.llm:
                raise ValueError("LLM is required for summary buffer memory")
                
            self.memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                return_messages=True,
                memory_key="chat_history",
                output_key="answer",
                max_token_limit=self.max_token_limit
            )
            
        elif self.memory_type == "redis":
            if not self.redis_url:
                raise ValueError("Redis URL is required for redis memory")
                
            message_history = RedisChatMessageHistory(
                session_id=session_id,
                url=self.redis_url
            )
            
            self.memory = ConversationBufferMemory(
                chat_memory=message_history,
                return_messages=True,
                memory_key="chat_history",
                output_key="answer"
            )
        
        else:
            raise ValueError(f"Unsupported memory type: {self.memory_type}")
            
        return self.memory
        
    def get_memory_dict(self) -> Dict[str, Any]:
        """
        Get memory configuration as a dictionary.
        
        Returns:
            Dictionary with memory configuration
        """
        memory_info = {
            "type": self.memory_type,
            "max_token_limit": self.max_token_limit if hasattr(self, "max_token_limit") else None
        }
        return memory_info


class DocChat:
    """Handles document-based chat interactions with enhanced memory."""
    
    def __init__(self, groq_api_key: str, retriever, model_name: str = "qwen-qwq-32b", 
                 memory_type: str = "summary_buffer", redis_url: Optional[str] = None,
                 session_id: str = "default"):
        """
        Initialize DocChat with document retriever, model parameters, and memory.
        
        Args:
            groq_api_key: API key for Groq
            retriever: Document retriever for fetching relevant content
            model_name: Model name to use
            memory_type: Type of memory to use
            redis_url: URL for Redis connection (required for redis memory)
            session_id: Unique identifier for the chat session
        """
        self.retriever = retriever
        self.llm = None
        self.qa_chain = None
        self.chat_history = []
        self.prompts = self._create_prompts()
        self.initialize_groq_components(groq_api_key, model_name)
        
        # Initialize memory
        self.memory_manager = MemoryManager(
            memory_type=memory_type,
            redis_url=redis_url,
            llm=self.llm
        )
        self.memory = self.memory_manager.create_memory(session_id)
        self.session_id = session_id
        
        # Initialize enhanced QA chain with memory
        self._initialize_qa_chain_with_memory()

    def _create_prompts(self) -> Dict:
        """
        Create prompt templates for different query types.
        
        Returns:
            Dictionary of prompt templates
        """
        # Base system message template for all queries
        base_system_template = """You are MedBot, an advanced AI assistant specialized in medical knowledge.

CAPABILITIES:
- Provide accurate, evidence-based medical information
- Explain medical terminology in clear, accessible language
- Interpret symptoms and medical conditions with precision
- Reference medical literature and guidelines when appropriate

CONSTRAINTS:
- You are NOT a replacement for professional medical diagnosis or treatment
- Always clarify that users should consult healthcare providers for personal medical advice
- Maintain strict medical accuracy - if unsure, acknowledge limitations
- Avoid making definitive diagnostic statements

RESPONSE FORMAT:
- Begin with a clear, direct answer to the question
- Provide context and additional relevant information
- Use bullet points for symptoms, treatments, or key facts when appropriate
- Include brief mention of relevant medical guidelines or consensus when applicable
- End with a reminder about consulting healthcare professionals when appropriate

CONVERSATION HISTORY:
{chat_history}

DOCUMENT CONTEXT BELOW:
{context}

Remember: Base your responses on the document context provided, while maintaining continuity with the conversation history. If the context doesn't contain relevant information, acknowledge this limitation.
"""

        # Human template focusing on the question
        human_template = """
{input}
"""

        # Create a ChatPromptTemplate
        base_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(base_system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

        # Create specialized templates for different query types
        prompts = {
            "general": base_prompt,
            
            "medical": ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(base_system_template + """
ADDITIONAL MEDICAL INSTRUCTIONS:
- Use proper medical terminology with layperson explanations
- Include ICD codes when identifying specific conditions
- Reference standard treatment protocols when applicable
- Mention both conventional and evidence-based alternative approaches
- Clarify levels of evidence (e.g., RCT, meta-analysis, case studies)
"""),
                HumanMessagePromptTemplate.from_template(human_template)
            ]),
            
            "educational": ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(base_system_template + """
EDUCATIONAL FOCUS:
- Structure responses with clear learning objectives
- Define all medical terms when first introduced
- Build explanations from basic concepts to more complex ideas
- Use anatomical references and physiological processes to explain mechanisms
- Incorporate mnemonics or memory aids when helpful
"""),
                HumanMessagePromptTemplate.from_template(human_template)
            ]),
            
            "detailed": ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(base_system_template + """
DETAILED ANALYSIS REQUIREMENTS:
- Provide in-depth coverage of the topic with subsections
- Include epidemiological data when relevant
- Discuss pathophysiology in detail
- Cover differential diagnosis considerations
- Elaborate on diagnostic criteria and testing modalities
- Detail treatment approaches with medication classes/options
- Address prognosis and complications
"""),
                HumanMessagePromptTemplate.from_template(human_template)
            ])
        }
        
        return prompts

    def initialize_groq_components(self, groq_api_key: str, model_name: str) -> None:
        """
        Initialize Groq chat components.
        
        Args:
            groq_api_key: API key for Groq
            model_name: Model name to use
        """
        try:
            self.llm = ChatGroq(
                api_key=groq_api_key,
                model_name=model_name,
                temperature=0.2,  # Lower temperature for more consistent medical responses
                max_tokens=2048    # Ensure sufficient tokens for detailed answers
            )
        except Exception as e:
            raise Exception(f"Failed to initialize Groq components: {str(e)}")

    def _initialize_qa_chain_with_memory(self) -> None:
        """Initialize document-based retrieval and answering chain with memory support."""
        try:
            # Create the question answering chain with memory integration
            question_answer_chain = create_stuff_documents_chain(
                self.llm, 
                self.prompts["general"],
                document_variable_name="context"
            )
            
            # Create the retrieval chain
            retrieval_chain = create_retrieval_chain(
                self.retriever, question_answer_chain
            )
            
            # Wrap the chain with message history capability
            self.qa_chain = RunnableWithMessageHistory(
                retrieval_chain,
                lambda session_id: self.memory,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
            
        except Exception as e:
            raise Exception(f"Failed to initialize QA chain with memory: {str(e)}")

    def query(self, question: str, query_type: str = "general") -> Dict:
        """
        Execute chat query with memory integration.
        
        Args:
            question: User question
            query_type: Type of query (medical, educational, detailed, or general)
            
        Returns:
            Dict: Response from the QA chain
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Make sure to initialize components first.")
        
        # Verify the query type exists or default to general
        if query_type not in self.prompts:
            print(f"Warning: Query type '{query_type}' not found. Using 'general' instead.")
            query_type = "general"
        
        try:
            # Create a specialized chain for this query type
            specialized_qa_chain = create_stuff_documents_chain(
                self.llm, 
                self.prompts[query_type],
                document_variable_name="context"
            )
            
            specialized_retrieval_chain = create_retrieval_chain(
                self.retriever, specialized_qa_chain
            )
            
            # Wrap the specialized chain with message history capability
            specialized_chain_with_memory = RunnableWithMessageHistory(
                specialized_retrieval_chain,
                lambda session_id: self.memory,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
            
            # Execute the query with memory integration
            response = specialized_chain_with_memory.invoke(
                {"input": question},
                config={"configurable": {"session_id": self.session_id}}
            )
            
            # Add metadata to the response
            response["query_type"] = query_type
            response["timestamp"] = datetime.now().isoformat()
            
            return response
        except Exception as e:
            error_msg = f"Error executing query: {str(e)}"
            print(error_msg)
            return {"answer": error_msg, "error": True, "query_type": query_type}

    def get_memory_contents(self) -> Dict:
        """
        Get the current contents of the memory.
        
        Returns:
            Dict containing memory variables
        """
        return self.memory.load_memory_variables({})

    def reset_memory(self) -> None:
        """Reset the memory to clear conversation history."""
        self.memory.clear()

    def add_context(self, context: str, role: str = "system") -> None:
        """
        Add additional context to memory.
        
        Args:
            context: Additional context to consider
            role: Role of the message (system, human, ai)
        """
        # Convert role to message type
        if role == "system":
            message = SystemMessage(content=context)
        elif role == "human":
            message = HumanMessage(content=context)
        elif role == "ai":
            message = AIMessage(content=context)
        else:
            raise ValueError(f"Invalid role: {role}. Use 'system', 'human', or 'ai'.")
        
        # Add message to memory
        self.memory.chat_memory.add_message(message)


class MedBotApp:
    """Main application class for MedBot with enhanced memory capabilities."""
    
    def __init__(self, data_dir: str = "Data/"):
        """
        Initialize MedBot application.
        
        Args:
            data_dir: Directory containing medical PDF documents
        """
        self.data_dir = data_dir
        self.config = ConfigManager()
        self.doc_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager()
        self.faiss_manager = None
        self.doc_chat = None
    
    def setup(self, model_name: str = "qwen-qwq-32b", create_new_index: bool = False,
              memory_type: str = "summary_buffer", session_id: str = "default") -> None:
        """
        Set up the MedBot application with memory support.
        
        Args:
            model_name: Model name to use for Groq
            create_new_index: Whether to create a new FAISS index
            memory_type: Type of memory to use
            session_id: Unique identifier for the chat session
        """
        # Load and process documents
        documents = self.doc_processor.load_pdf_documents(self.data_dir)
        text_chunks = self.doc_processor.split_documents(documents)
        print(f"Processed {len(text_chunks)} text chunks from documents")
        
        # Load embeddings
        embeddings = self.embedding_manager.load_embeddings()
        
        # Set up FAISS
        self.faiss_manager = FAISSManager(embeddings=embeddings)
        
        if create_new_index or not os.path.exists(self.faiss_manager.index_path):
            print("Creating new FAISS index...")
            self.faiss_manager.create_index(text_chunks)
        else:
            print("Loading existing FAISS index...")
            self.faiss_manager.load_index()
        
        # Create retriever
        retriever = self.faiss_manager.get_retriever()
        
        # Set up DocChat with memory
        self.doc_chat = self.create_doc_chat(
            retriever, 
            model_name, 
            memory_type, 
            session_id
        )
    
    def create_doc_chat(self, retriever, model_name: str = "qwen-qwq-32b",
                         memory_type: str = "summary_buffer", session_id: str = "default") -> Optional[DocChat]:
        """
        Create and return a DocChat instance with memory integration.
        
        Args:
            retriever: Document retriever
            model_name: Model name to use
            memory_type: Type of memory to use
            session_id: Unique identifier for the chat session
            
        Returns:
            DocChat instance or None if initialization fails
        """
        try:
            doc_chat = DocChat(
                groq_api_key=self.config.groq_api_key,
                retriever=retriever,
                model_name=model_name,
                memory_type=memory_type,
                redis_url=self.config.redis_url,
                session_id=session_id
            )
            return doc_chat
        except Exception as e:
            print(f"Failed to create DocChat: {str(e)}")
            return None
    
    def query(self, question: str, query_type: str = "medical") -> Dict:
        """
        Make a query to the DocChat.
        
        Args:
            question: Question to ask
            query_type: Type of query (medical, educational, detailed, or general)
            
        Returns:
            Response from the DocChat
        """
        if not self.doc_chat:
            raise ValueError("DocChat not initialized. Set up the application first.")
        
        return self.doc_chat.query(question, query_type)
    
    def get_memory_status(self) -> Dict:
        """
        Get information about the current memory status.
        
        Returns:
            Dict with memory information
        """
        if not self.doc_chat:
            raise ValueError("DocChat not initialized. Set up the application first.")
            
        memory_info = self.doc_chat.memory_manager.get_memory_dict()
        memory_contents = self.doc_chat.get_memory_contents()
        
        return {
            "configuration": memory_info,
            "contents": memory_contents,
            "session_id": self.doc_chat.session_id
        }
    
    def reset_memory(self) -> None:
        """Reset the current memory."""
        if not self.doc_chat:
            raise ValueError("DocChat not initialized. Set up the application first.")
            
        self.doc_chat.reset_memory()
        print(f"Memory for session '{self.doc_chat.session_id}' has been reset.")