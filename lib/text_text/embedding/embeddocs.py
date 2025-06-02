# embed different documents in different collections, in same directory (? is this right)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter #use AITextSplitter instead
from .agentic_chunker.agentic_chunker import LLMChunker
from langchain.schema import Document 

embedfn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #can vary model name

content = []
with open("pencilwiki.txt","r",encoding="utf-8") as file: 
    content.append(file.read(20))
    content.append(file.read(20))

docs = [Document(page_content=doc) for doc in content]
splitter = RecursiveCharacterTextSplitter(chunk_size=5, chunk_overlap=2) #can try to vary params
splitter2 = LLMChunker()
chunks = splitter.split_documents(documents=docs) #use split_text if splitter2
vectorstore = Chroma.from_documents(chunks, collection_name="mycoll", embedding=embedfn, persist_directory="db") 
#Chroma.add_documents for adding documents
#can have different collections in same directory

