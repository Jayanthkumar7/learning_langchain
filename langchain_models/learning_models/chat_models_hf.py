from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "google/gemma-4-31B-it",
    task="text-generation"
)

model = ChatHuggingFace(llm =llm )

result = model.invoke("who is the cheif minister of tamilnadu ?")
print(result.content)
