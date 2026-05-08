# ============================================================
# Phase 1 - Vector Based Persona Matching (The Router)
# Developer: Sachu Retna S M
# Grid07 Platform
# ============================================================

# ------------------------------------------------------------
# STEP 1: Import Libraries
# ------------------------------------------------------------
# You need:
# - chromadb (vector database)
# - sentence_transformers (to convert text to embeddings)
# - os and dotenv (for environment variables)


import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import os


# ------------------------------------------------------------
# STEP 2: Load Environment Variables
# ------------------------------------------------------------
# Load your .env file using dotenv

load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")




# ------------------------------------------------------------
# STEP 3: Initialize Embedding Model
# ------------------------------------------------------------
# Use SentenceTransformer with 'all-MiniLM-L6-v2' model
# This converts text → vector numbers

EMBED_MODEL = "all-MiniLM-L6-v2"

_model = None 
def get_embedding_function():
    global _model
    if _model is None:
        _model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    return _model

embedding_func = get_embedding_function()



# ------------------------------------------------------------
# STEP 4: Define Bot Personas
# ------------------------------------------------------------
# Create a dictionary with 3 bot personas
# Keys: "bot_a", "bot_b", "bot_c"
# Values: their personality description strings

personas = {
    "bot_a": "I believe AI and crypto will solve all human problems. I am highly optimistic about technology, Elon Musk, and space exploration. I dismiss regulatory concerns.",
    "bot_b": "I believe late-stage capitalism and tech monopolies are destroying society. I am highly critical of AI, social media, and billionaires. I value privacy and nature.",
    "bot_c": "I strictly care about markets, interest rates, trading algorithms, and making money. I speak in finance jargon and view everything through the lens of ROI."
}


# ------------------------------------------------------------
# STEP 5: Setup ChromaDB
# ------------------------------------------------------------
# Create a chromadb client
# Create a collection called "bot_personas"
_client = None

def get_chroma_client():
    global _client
    if _client is None:
        _client = chromadb.Client()
    return _client  
    
chroma_client = get_chroma_client()

collection_name = "bot_personas"

_collection = None
def get_collection():
    global _collection
    if _collection is None:
        _collection = chroma_client.create_collection(name=collection_name, embedding_function=embedding_func)
    return _collection

collection = get_collection()




# ------------------------------------------------------------
# STEP 6: Add Personas to ChromaDB
# ------------------------------------------------------------
# Loop through personas dictionary
# For each persona:
#   - Add the persona text as document
#   - Use the key (bot_a, bot_b, bot_c) as id

for bot_id, persona_text in personas.items():
    collection.add(
        documents=[persona_text],  
        ids= [bot_id]       
    )



# ------------------------------------------------------------
# STEP 7: Write the Routing Function
# ------------------------------------------------------------
# Function: route_post_to_bots(post_content, threshold=0.85)
#
# What it should do:
#   1. Take a post string as input
#   2. Query ChromaDB with that post
#   3. Get similarity scores for all 3 bots
#   4. Return only bots where similarity > threshold
#
# HINT for cosine distance → similarity conversion:
#   ChromaDB returns "distance" not "similarity"
#   similarity = 1 - distance
#
# HINT for querying:
#   results = collection.query(
#       query_texts=[post_content],
#       n_results=3,
#       include=["distances"]
#   )
threshold = 0.20 #fix: lowered threshold to 0.20 because all-MiniLM-L6-v2 similarity scores range between 0.2-0.5, default 0.85 was too strict and returning empty matche

def route_post_to_bots(post_content: str, threshold: float = threshold) -> list:
    """
    Routes a post to relevant bots based on persona similarity.
    
    Args:
        post_content: The text of the incoming post
        threshold: Minimum similarity score (default 0.85)
    
    Returns:
        List of matching bot ids
    """
    # Write your logic here:
    # 1. Query chromadb
    results = collection.query(
        query_texts= [post_content],      
        n_results=3,
        include=["distances"]
    )

    # 2. Extract ids and distances
    ids = results["ids"][0]
    distances = results["distances"][0]

    # 3. Convert distance to similarity and filter
    matched = []
    for bot_id, distance in zip(ids, distances):
        similarity = 1 - distance       # 1 - distance
        print(f"Bot: {bot_id}, Similarity: {similarity:.4f}")  # Debug print
        if similarity > threshold:    # check threshold
            matched.append(bot_id)

    return matched




# ------------------------------------------------------------
# STEP 8: Test Your Function
# ------------------------------------------------------------
# Test with these 3 posts and see which bots match

if __name__ == "__main__":
    
    test_posts = [
        "OpenAI just released a new model that might replace junior developers.",
        "Bitcoin hits new all-time high amid regulatory ETF approvals.",
        "Big tech companies are buying politicians and destroying democracy."
    ]
    
    for post in test_posts:
        print(f"\nPost: {post}")
        matched_bots = route_post_to_bots(post)
        print(f"Matched Bots: {matched_bots}")