from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer #can use tokenier directly from sentence-transformers
from langchain_ollama import ChatOllama
import re

# can be refined further to add overlaps + token limit  
# + prompt engineering + batch processing (currently sending entire doc and das why taking really long)

class LLMChunker:

    def __init__(self, ollama_model="llama3.2", embed_fn="sentence-transformers/all-MiniLM-L6-v2"):
        
        self.tokenizer = AutoTokenizer.from_pretrained(embed_fn)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0) # removing chunk overlap
        self.llm = ChatOllama(model=ollama_model, temperature=0) #use model with long context windows

    def split_text(self, document): # sentence-transformers embedding tend to have token limit 512?

        small_chunks = self.splitter.split_text(document)
        print(f"Created {len(small_chunks)} initial chunks")
        
        final_chunks = []
        chunked_doc = f"<|start_chunk_1|>"

        for i in range(1,len(small_chunks)):
            chunked_doc+= small_chunks[i-1]
            chunked_doc+=f"<|end_chunk_i|>\t<|start_chunk_i+1|>"


                
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are an assistant specialized in splitting text into thematically consistent sections. "
                    "The text has been divided into chunks, each marked with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number. "
                    "Your task is to identify the points where splits should occur, such that consecutive chunks of similar themes stay together. "
                    "Respond with a list of chunk IDs where you believe a split should be made. For example, if chunks 1 and 2 belong together but chunk 3 starts a new topic, you would suggest a split after chunk 2. THE CHUNKS MUST BE IN ASCENDING ORDER."
                    "Your response should be in the form: 'split_after: 3, 5'."
                    "Make sure the chunks you form are not too large (~2000 characters or ~500 tokens)."
                    "You can have slight chunk overlap if necessary." #?
                )
            },
            {
                "role": "user", 
                "content": (
                    "CHUNKED_TEXT: " + chunked_doc + "\n\n"
                    "Respond only with the IDs of the chunks after which you believe a split should occur. A split after the last chunk is invalid. YOU MUST RESPOND WITH AT LEAST ONE SPLIT. THESE SPLITS MUST BE IN ASCENDING ORDER.)"
                )
            },
        ]
                
        try:
            llm_response = self.llm.invoke(messages).content
            numbers = re.findall(r'\d+', llm_response) #should be in ascending order ideally #should add a check?
        
            #creating final chunks

            for i in numbers:
                i = int(i)
                if i != len(small_chunks):
                    #safety check
                    j = 0
                    temp_chunk = ""
                    while j<=i:
                        temp_chunk += small_chunks[j]
                    j = i+1
                final_chunks.append(temp_chunk)
            last_chunk = ''
            i = i+1
            while i < len(small_chunks)-1:
                last_chunk += small_chunks[i]
                i += 1
            final_chunks.append(last_chunk)

        except:
            #if llm.invoke fails? revert to RecursiveSplitter
            final_chunks = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50).split_text(document)

        print(f"Created {len(final_chunks)} semantic chunks") #chunks without any overlap if llmchunker
        return final_chunks
    
'''
with open("pencilwiki.txt","r", encoding="utf-8") as file:
    content = file.read(10000)

chunker = LLMChunker()
chunks = chunker.split_text(content)

with open("chunked_text.txt","w", encoding="utf-8") as file:
    file.writelines([i+"\n\nendofchunk\n\n" for i in chunks])
''' # a test 
