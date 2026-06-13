from langchain_community.document_loaders import TextLoader


loader = TextLoader(file_path="docs/cricket.txt" , encoding="utf-8")


content = loader.load()

print(content[0].page_content)
print(content[0].metadata)
