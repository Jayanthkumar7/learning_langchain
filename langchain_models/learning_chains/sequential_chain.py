from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")


prompt1 = PromptTemplate(
    template="Generate a detailed report on th topic {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template = "Generate a 5 line pointer summary from the text : {text}",
    input_variables=["text"]
)


parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic":"Growth of woman cricket in india"})

print(result)