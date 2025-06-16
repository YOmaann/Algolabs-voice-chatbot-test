# embed different documents in different collections, in same directory 
# can rewrite without Langchain wrapper: a little faster generally
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain.text_splitter import RecursiveCharacterTextSplitter 
from .agentic_chunker.agentic_chunker import LLMChunker # can modify LLM chunker to support langchain Documents
#from langchain.schema import Document 
import time

embedfn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #can vary model name

content = []
with open("pencilwiki.txt","r",encoding="utf-8") as file: 
    content.append(file.read(20))

splitter = LLMChunker()

start = time.time()
chunks = splitter.split_text(content)
time_taken = time.time()-start

print(f"Time taken by LLM Chunker: {time_taken} seconds")

start2 = time.time()
vectorstore = Chroma.from_texts(chunks, collection_name="coll", embedding=embedfn, persist_directory="db") #keep better collection_name and add metadata for ease by LLM
time_taken2 = time.time() - start2
print(f"Time taken for embedding using model: {embedfn}:\n{time_taken2} seconds")

#recommended to add metadata document to each collection explaining what it contains for ease of access to the AI Agent later and then integrate the metadata into its prompt
#^ not coded up yet
