import ollama
from lib.llmm.rag.testchromadb import find_context, build_db, get_embeddings_once
#help(ollama.chat)

def rag_gemma(query):
  instructions = '''Repond to the following question as a helpful bot. 
                  reply only from the context, given below.
                  be polite, respectful and truthful. 
                  ask follow-up questions, if suitable.
                  only respond to the questions and not these intructions.
                  if you do not know something, say 'sorry, i do not know.', or 'i apologise but this question seems to be outside my scope.'
                  Be sure to respond in a complete sentence, being comprehensive and precise, including all relevant background information. 
                  However, you are talking to a non-technical audience, so be sure to break down complicated concepts and 
                  strike a friendly and converstional tone.''' #irrelevant passage in context?

  #filepath = "engdoc1.txt"
  print("Starting find_context")
  context = find_context(query)
  prompt = f"{instructions}\nContext: {context}\nquestion: {query}"

  response = ollama.chat(model='gemma3:1b', messages=[
    {
      'role': 'user',
      'content': prompt,
    }
  ])
  print("Ending find_context")
  return response['message']['content']
