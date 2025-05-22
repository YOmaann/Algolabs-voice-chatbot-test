from google import genai
from google.genai import types

APIkey = "AIzaSyBRbh0JTGUMSOgZ2QRpA056qyrMinkhnsw"

client = genai.Client(api_key=APIkey)

'''for m in client.models.list():
    if "embedContent" in m.supported_actions:
        print(m.name)'''
'''models/embedding-001
models/text-embedding-004
models/gemini-embedding-exp-03-07
models/gemini-embedding-exp''' #models that support embedding

from chromadb import Documents, EmbeddingFunction, Embeddings

class GeminiEmbeddingFunction(EmbeddingFunction):
    # Specify whether to generate embeddings for documents, or queries
    def __init__(self):
        self.document_mode = True

    def __call__(self, input: Documents) -> Embeddings:
        if self.document_mode:
            embedding_task = "retrieval_document"
        else:
            embedding_task = "retrieval_query"

        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(
                task_type=embedding_task,
            ),
        )
        return [e.values for e in response.embeddings]
    

import chromadb

def find_context(query, documents):
    DB_NAME = "mydb"

    embed_fn = GeminiEmbeddingFunction()
    embed_fn.document_mode = True

    chroma_client = chromadb.Client()
    db = chroma_client.get_or_create_collection(name=DB_NAME, embedding_function=embed_fn)

    db.add(documents=documents, ids=[str(i) for i in range(len(documents))])

    #print(db.peek(1))

    # Switch to query mode when generating embeddings.
    embed_fn.document_mode = False

    result = db.query(query_texts=[query], n_results=1)["documents"]
    return(result)


