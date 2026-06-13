# using gemini cause this is free
# LLM's are being deprecated so the code is not written and proceding with ChatModels 

# OPENAI and ANthropic keys need to be purchased

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature = 2
)

result = model.invoke("what is the capital of india")

print(result.content)