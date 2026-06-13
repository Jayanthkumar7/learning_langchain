from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

parser  = JsonOutputParser()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)


template = PromptTemplate(
    template="Imagine a {type} city and give out key details like city name , king name , origin , location {format_instruction}",
    
    input_variables=["type"],
    partial_variables={"format_instruction":parser.get_format_instructions()}                        
                        )

chain = template | model | parser

result = chain.invoke({"type":input()})

print(result)