# import lib.models.gemini_helper as gemini
# import lib.llmm.llmprompting.gemmacall as gemmacall
from utils.lazy_import import import_local_module
from time import time

class llm_helper:
    def __init__(self, args, q, out_q, stop_e):
        self.args = args
        self.out_q = out_q
        self.q = q
        self.stop_e = stop_e # event to stop thread

        print("Preparing LLM..this might take a while..")

        if self.args.llm == 'gemini':
            gemini = import_local_module('lib/models/gemini_helper.py', 'gemini')
            self.fetch_response = gemini.fetch_response
        elif self.args.llm == 'gemma3':
            gemma3 = import_local_module('lib/llmm/llmprompting/gemmacall.py', 'gemma3')

            # Load embedding function
            ef = gemma3.get_embeddings_once()

            
            # build gemma3 database from the text. Might not need to call this
            client, col = gemma3.build_db(ef)
            self.fetch_response = gemma3.get_rag_gemma(ef, client, col)
        else:
            self.args.fetch_response = None

        print('Lets goo ::')

    def loop(self):
        while not self.stop_e.is_set():
            if self.q.empty():
                continue
        
            text = self.q.get()
            s_time = time()
            response = ""
            if self.fetch_response == None:
                print("No llm model selected bhai >_<")

            # print("gemma see")
            response = self.fetch_response(text)
            self.out_q.put(response)
            #elif self.args.llm == 'gemma3':
                #response = gemmacall("engdoc1.txt",text)
                
            e_time = time() - s_time
            print(f"llm latency = {e_time}s")
            print(f"llm >> {response}")
            
        
