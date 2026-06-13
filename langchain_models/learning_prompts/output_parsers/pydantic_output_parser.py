from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Initialize LLM
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)


# 1. Define the Child Model
class Kingdom(BaseModel):
    name: str = Field(description="the name of the kingdom / place / dystopian city")
    king_name: str = Field(description="The name of the person ruling the place / kingdom / city")
    history: str = Field(description="A 2 line short info about the city/kingdom/place")


# 2. Define the Parent Model
class Kingdoms(BaseModel):
    # Added Field structure properly and added context to the description
    title: str = Field(description="The exact theme provided by the user in the input.")
    list_kingdom: list[Kingdom] = Field(description="A list of fictional kingdoms belonging to the theme.")


# 3. Setup Parser
parser = PydanticOutputParser(pydantic_object=Kingdoms)


# 4. Define Prompt Template (Explicitly telling the model to match your input text)
template = PromptTemplate(
    template=(
        "You are a creative writer. Give me the key details of a fictional place under the theme: '{theme}'.\n"
        "Crucial Requirement: In your JSON response, set the value of the 'title' field to exactly '{theme}'.\n\n"
        "{format_instruction}"
    ),
    input_variables=["theme"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# 5. Execute Chain
chain = template | model | parser
result = chain.invoke({"theme": "mythological"})

# 6. Pretty Print JSON structure 
print(result.model_dump_json(indent = 2))
