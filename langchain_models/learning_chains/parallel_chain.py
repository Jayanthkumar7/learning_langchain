from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


load_dotenv()

model1 = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
model2 = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")


prompt1 = PromptTemplate(
    template = "generate short notes for the following topic \n {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="generate 5 quiz questions with 4 options to choose from for the topic {topic}",
    input_variables = ["topic"]
)

parser = StrOutputParser()
parallel_chain = RunnableParallel(
    {
        "notes":prompt1 | model1 | parser,
        "quiz":prompt2 | model2 | parser
    }
)

prompt3 = PromptTemplate(
    template=" merge the given notes and quiz for the respected topic one after the other \n notes -> {notes} \n quiz -> {quiz}",
    input_variables=["notes","quiz"]
)

merge_chain = prompt3 | model1 | parser


chain = parallel_chain | merge_chain

result = chain.invoke({"topic":"logistic regression"})

print(result)