from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    temperature=0
)

model = ChatHuggingFace(llm =llm)

parser = StrOutputParser()
template1 = PromptTemplate(
    template="Give a brief information about {topic}",
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="summarize the {text} in about 5 lines",
    input_variables=["text"]
)


chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic":"politicians and corruptions in india"})

print(result)