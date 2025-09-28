# 1. LangChain Document Loading
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader,
    UnstructuredPowerPointLoader
)
from langchain.schema import Document
from pathlib import Path
from typing import List
import os

def load_documents_langchain(directory_path: str) -> List[Document]:
    """Load documents using LangChain loaders"""
    documents = []
    directory = Path(directory_path)
    
    # Supported file types and their loaders
    loaders_map = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.pptx': UnstructuredPowerPointLoader
    }
    
    for file_path in directory.glob('**/*'):
        if file_path.is_file() and file_path.suffix.lower() in loaders_map:
            print(f"Loading: {file_path.name}")
            
            try:
                loader_class = loaders_map[file_path.suffix.lower()]
                loader = loader_class(str(file_path))
                file_docs = loader.load()
                
                # Add metadata
                for doc in file_docs:
                    doc.metadata.update({
                        'filename': file_path.name,
                        'file_type': file_path.suffix.lower(),
                        'source_path': str(file_path)
                    })
                
                documents.extend(file_docs)
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    print(f"Loaded {len(documents)} documents")
    return documents

# 2. LangChain Text Splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List

def chunk_documents_langchain(documents: List[Document]) -> List[Document]:
    """Chunk documents using LangChain text splitter"""
    
    # Use same settings as your existing approach
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        length_function=len,
    )
    
    # Split documents
    chunks = text_splitter.split_documents(documents)
    
    # Add chunk-specific metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata['chunk_id'] = i
        chunk.metadata['chunk_text'] = chunk.page_content[:100] + "..."
    
    print(f"Created {len(chunks)} chunks")
    return chunks

# 3. LangChain Pinecone Integration
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings  # You can switch to other embeddings
from pinecone import Pinecone
from langchain.schema import Document
from typing import List
import os

def create_vectorstore_langchain(
    chunks: List[Document], 
    index_name: str,
    pinecone_api_key: str,
    openai_api_key: str = None
) -> PineconeVectorStore:
    """Create Pinecone vector store using LangChain"""
    
    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)
    
    # Check if index exists, create if not
    if index_name not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI embeddings dimension
            metric='cosine'
        )
    
    # Initialize embeddings (you can use others like HuggingFace)
    if openai_api_key:
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    else:
        # Fallback: use your existing embeddings approach
        from embeddings_utils import upload_to_pinecone
        # Convert LangChain docs to your format
        docs_for_upload = []
        for chunk in chunks:
            docs_for_upload.append({
                "id": f"{chunk.metadata['filename']}_chunk_{chunk.metadata['chunk_id']}",
                "text": chunk.page_content,
                "metadata": chunk.metadata
            })
        upload_to_pinecone(docs_for_upload, index_name, pinecone_api_key)
        return None
    
    # Create vector store
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )
    
    print(f"Uploaded {len(chunks)} chunks to Pinecone")
    return vectorstore

# Alternative: Use existing vector store
def get_existing_vectorstore(index_name: str, pinecone_api_key: str, openai_api_key: str):
    """Connect to existing Pinecone index"""
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings
    )
    return vectorstore

# 4. LangChain MultiQuery Retrieval
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI  # or use custom LLM wrapper
from langchain.schema import BaseRetriever
import requests
import os

# Custom Perplexity LLM for MultiQuery (if not using OpenAI)
class PerplexityLLM:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.perplexity.ai/chat/completions"
    
    def invoke(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "sonar",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }
        
        response = requests.post(self.url, headers=headers, json=payload)
        return response.json()["choices"][0]["message"]["content"]

def create_multiquery_retriever(vectorstore, llm_type="openai"):
    """Create MultiQuery retriever with LangChain"""
    
    if llm_type == "openai" and os.getenv("OPENAI_API_KEY"):
        # Use OpenAI
        llm = ChatOpenAI(temperature=0)
    else:
        # Use Perplexity (limited support for MultiQuery)
        llm = PerplexityLLM(os.getenv("SONAR_API_KEY"))
    
    # Create MultiQuery retriever
    retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        llm=llm
    )
    
    return retriever

def retrieve_with_multiquery(retriever, query: str):
    """Retrieve documents using MultiQuery"""
    try:
        docs = retriever.get_relevant_documents(query)
        
        # Format results similar to your existing function
        results = []
        for doc in docs:
            results.append({
                'text': doc.page_content,
                'metadata': doc.metadata,
                'score': getattr(doc, 'score', 1.0)  # Score if available
            })
        
        return results
    except Exception as e:
        print(f"MultiQuery failed, falling back to basic retrieval: {e}")
        # Fallback to basic retrieval
        basic_docs = vectorstore.similarity_search(query, k=5)
        return [{'text': doc.page_content, 'metadata': doc.metadata} for doc in basic_docs]
    
# 5. LangChain Web Search (No Agents)
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

def setup_web_search():
    """Setup web search tool without agents"""
    search = DuckDuckGoSearchRun()
    return search

def web_search_if_needed(query: str, web_search_tool) -> str:
    """Use web search for current/recent information"""
    
    # Keywords that suggest need for current info
    current_keywords = ["current", "latest", "2025", "recent", "update", "new"]
    
    if any(keyword in query.lower() for keyword in current_keywords):
        try:
            print("Searching web for current information...")
            results = web_search_tool.run(query)
            return f"Current web information:\n{results}\n\n"
        except Exception as e:
            print(f"Web search failed: {e}")
            return ""
    
    return ""    

# 6. LangChain RAG Chat Pipeline
from typing import List, Dict
import requests
import os

def langchain_rag_chat(
    query: str, 
    retriever, 
    web_search_tool,
    conversation_history: List = None
) -> str:
    """Complete RAG pipeline using LangChain methods"""
    
    # 1. Check if web search needed and retrieve current info
    web_info = web_search_if_needed(query, web_search_tool)
    
    # 2. Retrieve relevant documents using MultiQuery
    retrieved_docs = retrieve_with_multiquery(retriever, query)
    
    # 3. Format context from retrieved documents
    context = "Relevant policy documents:\n"
    for i, doc in enumerate(retrieved_docs[:5], 1):
        filename = doc['metadata'].get('filename', 'Unknown')
        text_snippet = doc['text'][:300] + "..."
        context += f"\n[DOCUMENT {i}] From {filename}:\n{text_snippet}\n"
    
    # 4. Prepare conversation history
    history_context = ""
    if conversation_history:
        history_context = "\n\nConversation history:\n"
        for exchange in conversation_history[-3:]:  # Last 3 exchanges
            history_context += f"User: {exchange.get('user', '')}\n"
            history_context += f"Assistant: {exchange.get('assistant', '')[:200]}...\n"
    
    # 5. Create system prompt
    system_prompt = f"""You are a reproductive health policy compliance assistant. 
    Use the provided documents to answer questions accurately.
    
    {context}
    {web_info}
    {history_context}
    
    Instructions:
    - Prioritize information from the policy documents
    - Cite sources using [DOCUMENT X] format
    - If web information is relevant, mention it's current/recent
    - Be specific and actionable
    """
    
    # 6. Generate response using Perplexity/Sonar
    response = generate_rag_response(system_prompt, query)
    
    return response

def generate_rag_response(system_prompt: str, user_query: str) -> str:
    """Generate response using Sonar API"""
    headers = {
        "Authorization": f"Bearer {os.getenv('SONAR_API_KEY')}",
        "Content-Type": "application/json"
    }
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    payload = {
        "model": "sonar",
        "messages": messages,
        "temperature": 0.0
    }
    
    try:
        response = requests.post(
            "https://api.perplexity.ai/chat/completions", 
            headers=headers, 
            json=payload
        )
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error generating response: {str(e)}"
    
    # 7. Complete Pipeline Example
from dotenv import load_dotenv
import os

load_dotenv()

def run_complete_langchain_pipeline():
    """Complete document processing and RAG pipeline"""
    
    # Configuration
    directory_path = "Policy Pulse + AVE collab"
    index_name = "policypulse"
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")  # Optional
    
    print("Starting LangChain RAG Pipeline...")
    
    # Step 1: Load documents
    print("\n1. Loading documents...")
    documents = load_documents_langchain(directory_path)
    
    # Step 2: Chunk documents
    print("\n2. Chunking documents...")
    chunks = chunk_documents_langchain(documents)
    
    # Step 3: Create vector store (if needed)
    print("\n3. Setting up vector store...")
    if openai_api_key:
        vectorstore = create_vectorstore_langchain(
            chunks, index_name, pinecone_api_key, openai_api_key
        )
    else:
        # Use existing upload function
        create_vectorstore_langchain(chunks, index_name, pinecone_api_key)
        vectorstore = None
    
    # Step 4: Setup retrieval
    print("\n4. Setting up retrieval...")
    if vectorstore:
        retriever = create_multiquery_retriever(vectorstore)
    else:
        # Fallback: use your existing retrieval
        from ai_agent import retrieve_relevant_chunks
        class FallbackRetriever:
            def get_relevant_documents(self, query):
                results = retrieve_relevant_chunks(query, index_name, pinecone_api_key)
                return [{'text': r['text'], 'metadata': r.get('metadata', {})} for r in results]
        retriever = FallbackRetriever()
    
    # Step 5: Setup web search
    print("\n5. Setting up web search...")
    web_search_tool = setup_web_search()
    
    return retriever, web_search_tool

def interactive_langchain_rag():
    """Interactive chat using LangChain pipeline"""
    retriever, web_search_tool = run_complete_langchain_pipeline()
    conversation_history = []
    
    print("\n" + "="*60)
    print("LangChain Policy Pulse Assistant Ready!")
    print("Features: MultiQuery retrieval, Web search, Document analysis")
    print("Type 'quit' to exit")
    print("="*60)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        # Generate response
        response = langchain_rag_chat(
            user_input, 
            retriever, 
            web_search_tool,
            conversation_history
        )
        
        print(f"\nAssistant: {response}")
        
        # Update conversation history
        conversation_history.append({
            'user': user_input,
            'assistant': response
        })

# Run the pipeline
if __name__ == "__main__":
    interactive_langchain_rag()