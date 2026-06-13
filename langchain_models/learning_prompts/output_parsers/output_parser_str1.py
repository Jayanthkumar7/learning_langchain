from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm =llm)


t1 = PromptTemplate(
    template="explain about the {topic} in detailed.",
    
    input_variables=['topic']
                    )

t2 = PromptTemplate(template="Write a 5 line summary of the given text {text}" , input_variables=['text'])


p1 = t1.invoke({"topic":"Upcoming use of Quantum mechanics"})

print(p1.to_string())

result = model.invoke(p1)


p2 = t2.invoke({"text":result.content})

result1 = model.invoke(p2)

print(result1.content)
