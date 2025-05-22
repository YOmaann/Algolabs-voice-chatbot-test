import lib.models.gemini_helper as gemini
#import lib.llmm.llmprompting.gemmacall as gemmacall

class llm_helper:
    def __init__(self, args, q, out_q, stop_e):
        self.args = args
        self.out_q = out_q
        self.q = q
        self.stop_e = stop_e # event to stop thread

    def loop(self):
        while not self.stop_e.is_set():
            if self.q.empty():
                continue
        
            text = self.q.get()
            response = ""
            if self.args.llm == 'gemini':
                response = gemini.fetch_response(text)
                self.out_q.put(response)
            #elif self.args.llm == 'gemma3':
                #response = gemmacall("engdoc1.txt",text)

            else:
                print("No llm model selected bhai >_<")
                
            print(f"llm >> {response}")
        
