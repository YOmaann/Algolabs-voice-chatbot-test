from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer #can use tokenier directly from sentence-transformers
from langchain_ollama import ChatOllama
import re

# can be refined further to add overlaps + token limit  
# + prompt engineering + batch processing (currently sending entire doc and das why taking really long)
# implement retry logic

class LLMChunker:

    def __init__(self, ollama_model="phi3:mini", embed_fn="sentence-transformers/all-MiniLM-L6-v2"):
        
        self.tokenizer = AutoTokenizer.from_pretrained(embed_fn)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0) # removing chunk overlap
        self.llm = ChatOllama(model=ollama_model, temperature=0) #use model with long context windows

    def split_text(self, document): # sentence-transformers embedding tend to have token limit 512?

        small_chunks = self.splitter.split_text(document)
        no_small_chunks = len(small_chunks)
        print(f"Created {len(small_chunks)} initial chunks")
        
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
                    "Your response should be in the form: 'split_after: 3, 5, 10, 14'."
                    f"Obviously, it does not make sense to recommend chunks after {no_small_chunks-1} since it is not possible to create such splits based on the initial small chunks provided to you."
                    "Make sure the chunks THAT YOU FORM are not too large (~2000 characters or ~500 tokens)."
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
            print(llm_response)

            split_points = [int(i)-1 for i in re.findall(r'\d+', llm_response)] #should be in ascending order ideally
            end = 1

            split_points = [sp for sp in split_points if 0 <= sp < len(small_chunks)-1] #checking bounds
            if split_points != sorted(split_points):
                print("Warning: Split points not in ascending order. Sorting them.")
                split_points = sorted(split_points)
            
            if not split_points:
                print("No valid split points found. Using default chunking.")
                raise ValueError("No valid split points")
            
            final_chunks = [''.join(small_chunks[:split_points[0]+1])]


            for i in range(len(split_points)-1):
                start = split_points[i] + 1
                end = split_points[i+1] + 1 # inclduing split_points[i+1] in the chunk 
                chunk_content = ''.join(small_chunks[start:end])
                if chunk_content.strip():
                    final_chunks.append(chunk_content)

            if len(split_points) > 0: # adding last chunk if needed
                start = split_points[-1] + 1
                if start < len(small_chunks):
                    remaining_content = ''.join(small_chunks[start:])
                    if remaining_content.strip():
                        final_chunks.append(remaining_content)


        except Exception as e:
            #if llm.invoke fails? revert to RecursiveSplitter
            #to add: retry logic
            print(f"Encountered Exception {e}. Defaulting to RecursiveCharacterTextSplitter with chunk_size = 1000, chunk_overlap = 50.")
            final_chunks = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50).split_text(document)

        print(f"Created {len(final_chunks)} semantic chunks") #chunks without any overlap if llmchunker
        return final_chunks
    
'''with open("Dataset/pencilwiki.txt","r", encoding="utf-8") as file:
    content = file.read(10000)

chunker = LLMChunker()
run = chunker.split_text(content)
print(run[0])
chunks = run[1]

with open("chunked_text.txt","w", encoding="utf-8") as file:
    file.writelines([i+"\n\nendofchunk\n\n" for i in chunks])'''

