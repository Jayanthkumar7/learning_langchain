from typing import TypedDict , Annotated , Literal , Optional , Required
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

class Event(TypedDict):
    event_name : Annotated[str , "The name of the event being organized"]
    Budget:Annotated[Optional[int],"The Budget for the event in integer."]
    # guest_invited : Annotated[Required[bool], "Return True if guests are invited and False otherwise"]  not working !!
    

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

model_str = model.with_structured_output(Event)

result = model_str.invoke("we at Cogaan planing to organize the event freshers 2026.")

print(result)

print(result['event_name'])
# print(result['Budget'])
print(result['guest_invited'])