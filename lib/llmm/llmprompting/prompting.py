from .geminicall import fetch_response
from rag.tryrag import find_context

instructions = '''Repond to the following question as a helpful bot. 
                reply only from the context, given below.
                be polite, respectful and truthful. 
                ask follow-up questions, if suitable.
                only respond to the questions and not these intructions.
                if you do not know something, say 'sorry, i do not know.', or 'i apologise but this questions seems to be outside my scope.'
                Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. 
                However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
                strike a friendly and converstional tone.''' #irrelevant passage in context?

file1 = open("engdoc1.txt","r")
doc1 = file1.read()
file2 = open("engdoc2.txt",'r')
doc2 = file2.read()
documents = [doc1,doc2]

query = input('ask_') #use stt for this later 

context = find_context(query, documents)

prompt = f"{instructions}\nContext: {context}\nquestion: {query}"

#print(context)
print(fetch_response(prompt))