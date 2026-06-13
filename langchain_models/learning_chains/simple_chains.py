from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")


template = PromptTemplate(
    template='generate an email to {person} for the purpose "{purpose}"',
    input_variables=["person","purpose"]
)

parser = StrOutputParser()

chain = template | model | parser

result = chain.invoke({"person":input("enter the person:\n"),"purpose":input("enter the purpose of mail:\n")})

print(result)