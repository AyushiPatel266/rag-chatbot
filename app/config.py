from dotenv import load_dotenv
import os

load_dotenv()

# switched to Groq since it's free, works well for RAG
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# chunk size of 500 works better than 1000 in my tests
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# how many chunks to pull back per query
TOP_K = 4