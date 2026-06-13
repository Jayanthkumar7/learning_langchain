from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from transformers.utils import logging


logging.set_verbosity_error()
load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')


while True:
    user_input = input("You: ")
    if user_input == 'exit':
        break
    result = model.invoke(user_input)
    print(f"AI: {result.content}")
    
    