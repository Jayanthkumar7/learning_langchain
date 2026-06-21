import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from transformers.utils import logging


logging.set_verbosity_error()

doc1 = Document(
    page_content="""
    - Virat Kohli is the only player in the Indian Premier League (IPL) to have played all seasons for one team: Royal Challengers Bengaluru (RCB)

    - He was picked by RCB soon after he captained India to victory in the 2008 Under-19 World Cup and has been retained by them ever since

    - Kohli is the IPL's highest run-scorer and the only one with more than 8000 runs

    - He holds the record for most IPL centuries (eight) as well as most runs in a season (973 in 2016)

    - He won the Orange Cap twice, in 2016 and 2024
    """,
    metadata={"Team": "Royal Challengers Bengaluru"},
)

doc2 = Document(
    page_content="""
    - Rohit Sharma is India's most successful IPL captain, leading Mumbai Indians to five IPL titles

    - He is the only player to score three double centuries in One Day Internationals (ODIs)

    - Rohit captained India to the 2024 ICC T20 World Cup title

    - Known as the "Hitman", he is one of the most elegant stroke players in world cricket

    - He has scored more than 11,000 ODI runs and numerous international centuries
    """,
    metadata={"Team": "Mumbai Indians"},
)

doc3 = Document(
    page_content="""
    - MS Dhoni is regarded as one of the greatest captains in cricket history

    - He led India to victories in the 2007 ICC T20 World Cup, 2011 ICC Cricket World Cup, and 2013 Champions Trophy

    - Dhoni is famous for his calm demeanor and exceptional finishing ability

    - He has captained Chennai Super Kings to multiple IPL championships

    - He remains one of the most successful wicketkeeper-batters in international cricket
    """,
    metadata={"Team": "Chennai Super Kings"},
)

doc4 = Document(
    page_content="""
    - Sachin Tendulkar is widely known as the "God of Cricket"

    - He is the highest run-scorer in international cricket history

    - Sachin was the first player to score 100 international centuries

    - He played for India for 24 years from 1989 to 2013

    - He was a key member of India's 2011 Cricket World Cup-winning squad
    """,
    metadata={"Team": "Mumbai Indians (Former)"},
)

doc5 = Document(
    page_content="""
    - Jasprit Bumrah is regarded as one of the best fast bowlers of the modern era

    - He is known for his unique bowling action and deadly yorkers

    - Bumrah played a crucial role in India's overseas Test victories

    - He has consistently been among the top-ranked bowlers across formats

    - He has been a key match-winner for Mumbai Indians in the IPL
    """,
    metadata={"Team": "Mumbai Indians"},
)

doc6 = Document(
    page_content="""
    - Ravindra Jadeja is one of the finest all-rounders in world cricket

    - He is known for his accurate left-arm spin, explosive batting, and brilliant fielding

    - Jadeja was part of India's 2013 Champions Trophy-winning team

    - He has taken hundreds of international wickets and scored thousands of runs

    - He has been a vital player for Chennai Super Kings across multiple IPL title wins
    """,
    metadata={"Team": "Chennai Super Kings"},
)

doc7 = Document(
    page_content="""
    - Hardik Pandya is one of India's premier pace-bowling all-rounders

    - He played a major role in India's victories across ICC tournaments

    - Hardik captained Gujarat Titans to an IPL title in their debut season (2022)

    - He is known for his powerful hitting and ability to perform under pressure

    - His versatility makes him one of the most valuable white-ball cricketers
    """,
    metadata={"Team": "Mumbai Indians"},
)

docs = [doc1, doc2, doc3, doc4, doc5, doc6, doc7]
vector_store = Chroma(
    embedding_function=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
    persist_directory="chroma_db",
    collection_name="sample",
)

vector_store.add_documents(docs)

vector_store.get(include=["embeddings", "documents", "metadatas"])

ans_doc = vector_store.similarity_search(query="who plays for the franchise rcb ?", k=2)


print(ans_doc)
