from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser , ResponseSchema
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm =llm)


schema = [
    ResponseSchema(name="kingdom_name" , description="the name of the kingdom"),
    ResponseSchema(name="origin" , description="the origin of the kingdom"),
    ResponseSchema(name="king_name" , description="the king's name of the kingdom"),

]
parser = StructuredOutputParser.from_response_schemas(schema)


template = PromptTemplate(
    template="Imagine a city of {type} and give the key attributes \n {format_instructions}",
    input_variables=["type"],
    partial_variables={"format_instructions":parser.get_format_instructions()}
)



chain = template | model | parser

result = chain.invoke({"type":input("entr the type of city !")})

print(result)




