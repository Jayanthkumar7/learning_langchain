from langchain_community.document_loaders import Docx2txtLoader


loader = Docx2txtLoader(file_path="docs/thesis.docx")

docs = loader.load()

print(docs[0].page_content)
