from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# Use HuggingFaceEndpointEmbeddings for API-based generation
embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    # huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN") # Optional if in .env
)

text = "delhi is the capital of india !"

# This now makes a remote API call instead of a local calculation
vector = embedding.embed_query(text)

print(str(vector))

