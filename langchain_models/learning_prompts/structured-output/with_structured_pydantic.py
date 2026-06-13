from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel , EmailStr , Field
from typing import Optional
load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

class Student(BaseModel):
    name : str 
    age : Optional[str] = None
    Email : EmailStr
    cgpa : float =  Field(gt=-1 , lt=10 , default= 0 , description="A decimal value depicting the CGPA of the student")
    

model_str = model.with_structured_output(Student)

result = model_str.invoke("hi this is jayanth and my email is nallagatlajayanthkumar@gmail.com with a cgpa of 9.5")

print(result)