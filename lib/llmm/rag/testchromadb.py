import chromadb
from chromadb.utils import embedding_functions

def chunking(filepath): #.txt file in the form of Q: '' \n A: '' \n and so on
    documents = []
    ids=[]

    with open(filepath,"r") as file:
        content = file.readlines()

    j=1
    for i in range(len(content)): #chunking manually
        #temp = ''
        if content[i][0] == 'Q':
            temp = content[i]
        elif content[i][0] == 'A':
            temp += content[i]
            documents.append(temp)
            ids.append(str(j))
            j+=1
    return documents, ids
    
def find_context(filepath, query, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    
    documents,ids = chunking(filepath)

    client = chromadb.Client()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name) 
    coll = client.get_or_create_collection(name="testcoll", embedding_function=ef)

    coll.add(
        documents=documents, #can input metadata about each doc
        ids=ids,
    )

    results = coll.query(
        query_texts=[query],
        n_results=2,
        include=['documents'] #metadatas, ids, embeddings also
    )

    return results['documents']