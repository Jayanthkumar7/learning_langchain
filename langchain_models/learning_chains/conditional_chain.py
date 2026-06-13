from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 1. Define Pydantic Schema
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="the sentiment of the given feedback.")
    

parser_2 = PydanticOutputParser(pydantic_object=Feedback)
parser = StrOutputParser()

# 2. Define Prompts
prompt = PromptTemplate(
    template="classify the given feedback based on the sentiment it holds. \n feedback -> {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser_2.get_format_instructions()}
)

prompt2 = PromptTemplate(
    template="give a 2 line response for the user's feedback which was -> {sentiment}",
    input_variables=["sentiment"]
)

# 3. Fix the Branching Logic
# We ensure the lambda checks the Pydantic object property, 
# and then we pass a proper dictionary context down to prompt2.
classifier_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive',  prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative',  prompt2 | model | parser),
    RunnableLambda(lambda x: "invalid feedback as the input!") 
)

chain = prompt | model | parser_2 | classifier_chain

result = chain.invoke({"feedback": "the place stinks and the food quality is below par !"})

print(result)