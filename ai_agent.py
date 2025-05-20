from pinecone import Pinecone
import requests
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Get Pinecone API key from environment variable
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
SONAR_API_KEY = os.environ.get("SONAR_API_KEY")

def get_system_prompt(retrieved_chunks):
    system_prompt = (
        "You are an expert compliance assistant specializing in workplace reproductive and fertility health policies.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "- ONLY use information contained in the provided documents to answer questions\n"
        "- If the documents don't contain the answer, state clearly that you don't have that information\n"
        "- NEVER make up or hallucinate information not present in the documents\n"
        "- NEVER reference companies, monetary values, or details not explicitly mentioned in the documents\n"
        "- Provide specific citations linking each piece of information to its source document\n"
        "- When uncertain about any detail, express uncertainty rather than guessing\n\n"
        "Your role is to:\n"
        "- Provide accurate information based SOLELY on the provided context documents\n"
        "- Generate well-structured responses that clearly separate facts from the documents\n"
        "- Always cite your sources with clear document numbers\n"
        "- Refuse to speculate beyond what is explicitly stated in the documents\n"
        "- Prioritize searching official government sources, serious think tanks, research institutes and serious newspapers and magazines\n"
        "- Clearly list the primary sources used for the summary. You must include details like authors, and direct URL if available\n"
    )
    # Add retrieved context to the system prompt
    if retrieved_chunks:
        system_prompt += "\n\n===== REFERENCE DOCUMENTS =====\n\n"
        for i, chunk in enumerate(retrieved_chunks):
            system_prompt += f"DOCUMENT {i+1}: [ID: {chunk['id']}]\n{chunk['text']}\n\n"
    else:
        system_prompt += "\n\nNO REFERENCE DOCUMENTS AVAILABLE. If you cannot answer the query based on your general knowledge, please state that you do not have sufficient information to provide an accurate answer."
    
    return system_prompt

def retrieve_relevant_chunks(text, index_name, api_key, top_k=5):
    """
    Retrieve relevant text chunks from Pinecone based on a query
    
    Parameters:
    text (str): The user's query text
    index_name (str): Name of Pinecone index
    api_key (str): Pinecone API key
    top_k (int): Number of relevant chunks to retrieve
    
    Returns:
    list: List of relevant document chunks
    """
    from pinecone import Pinecone
    
    # Initialize Pinecone
    pc = Pinecone(api_key=api_key)
    
    # Get the index
    index = pc.Index(index_name)
 
    # Use the search method with proper format
    response = index.search(
        namespace="",
        query={
            "inputs": {"text": text},
            "top_k": top_k
        },
        fields=["ID", "text", "metadata"],  # Specify the fields you want to retrieve
    )
        
    print(f"Search response: {response}")
    
    # Extract and return the retrieved chunks
    retrieved_chunks = []

    # Force it into a dict if needed
    if not isinstance(response, dict) and hasattr(response, "to_dict"):
        response = response.to_dict()

    hits = response.get("result", {}).get("hits", [])
    print(f"Found {len(hits)} hits")

    retrieved_chunks = []
    for hit in hits:
        retrieved_chunks.append({
            "id":       hit.get("_id", "unknown"),
            "score":    hit.get("_score", 0),
            "text":     hit.get("fields", {}).get("text", ""),
            "metadata": hit.get("metadata", {})  # if you stored metadata
        })
    return retrieved_chunks
        
    
    
def query_sonar(system_prompt, user_query):
    """
    Query the SONAR API with system prompt and user query
    
    Parameters:
    system_prompt (str): System prompt including retrieved context
    user_query (str): User's question
    
    Returns:
    str: Response from SONAR
    """
    
    
    # SONAR API endpoint and key (replace with your actual values)
    SONAR_API_URL = "https://api.perplexity.ai/chat/completions"  # Replace with actual endpoint
    SONAR_API_KEY = os.environ.get("SONAR_API_KEY")
    
    # Prepare the request
    headers = {
        "Authorization": f"Bearer {SONAR_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "sonar",  # Replace with appropriate model name
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        "temperature": 0.0,  # Lower temperature for a more deterministic response
        #"search_domain_filter": ["wikipedia.org", "docs.python.org"],
        "web_search_options": {
            "search_context_size": "low"
        }
    }
    # Make the API call
    response = requests.post(SONAR_API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error calling SONAR API: {response.status_code} - {response.text}"
    

def answer_question(user_query, index_name, api_key=PINECONE_API_KEY, top_k=5):
    """
    Main function to answer a user question using RAG approach
    
    Parameters:
    user_query (str): User's question
    index_name (str): Name of Pinecone index
    api_key (str): Pinecone API key
    top_k (int): Number of relevant chunks to retrieve
    
    Returns:
    str: Response from SONAR
    """
    # Retrieve relevant chunks
    print(f"Retrieving relevant chunks for query: {user_query}")
    retrieved_chunks = retrieve_relevant_chunks(user_query, index_name, api_key, top_k)
    print(f"Retrieved {len(retrieved_chunks)} relevant chunks")
    
    # Generate system prompt with retrieved chunks
    system_prompt = get_system_prompt(retrieved_chunks)
    
    # Query SONAR API
    print("Querying SONAR API...")
    response = query_sonar(system_prompt, user_query)
    
    return response

def interactive_qa(index_name="policypulse", api_key = PINECONE_API_KEY):
    """
    Interactive Q&A loop for testing the RAG system
    """
    print("Welcome to Policy Pulse, the Reproductive and Fertility Health Compliance Agent!")
    print("Type 'exit' to quit.")
    
    while True:
        user_query = input("\nYour question: ")
        
        if user_query.lower() in ["exit", "quit", "bye"]:
            print("Thank you for using the Compliance Bot. Goodbye!")
            break
        
        # Process the query
        response = answer_question(user_query, index_name, api_key)
        
        # Display the response
        print("\nCompliance Bot Response:")
        print(response)

if __name__ == "__main__":      
    
    # Start the interactive Q&A session
    interactive_qa()