# view embeddocs.py for embedding a document

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import time


embedfn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",  model_kwargs={"device": "cuda:0"})
vectorstore = Chroma(collection_name="mycoll", embedding_function=embedfn, persist_directory="db", collection_metadata={"hnsw:space": "cosine"})
#using cosine similarity

def find_context(query, collname="mycoll", directory="db", no_of_docs=5):
    #finding context via embeddings already stored in directory
    results = vectorstore.similarity_search(query, no_of_docs) 
    return results

def askLLM(model_name, query, context, temp=0, top_k=None, top_p=None, num_predict=200):
    template = ChatPromptTemplate.from_messages([
        ("system",'''You are a Customer Support AI Bot. Follow these instructions: 
        {instructions}.
        
        Here is the relevant context for answering questions:
        {context}'''),
        
        ("user", "{question}")
    ])

    instructions = "be nice. answer from context only. ignore irrelevant context."
    #context = "pencils are round."
    messages = template.format_messages(instructions=instructions, context=context, question=query)

    llm = ChatOllama(
        model=model_name,
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
        num_predict=num_predict)    

    response = llm.invoke(messages)
    return response

if __name__ == "__main__":

    while True:
        query = input('Query>>')
        sT = time.time()
        context = find_context(query, no_of_docs=10)
        answer = askLLM("llama3.2", query, context).content
        print(answer)
        print(f"Time : {time.time() - sT}")
