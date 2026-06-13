from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.messages import SystemMessage , AIMessage , HumanMessage
from dotenv import load_dotenv

load_dotenv()


model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

chat_history = [
    SystemMessage(content="your name is BUFF , a bot expert in cracking jokes.")
]

while True:
    user_input = input("You : ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI : ",result.content)
    


print()

print(chat_history)

print()



