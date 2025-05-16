import lib.models.gemini_helper as gemini


class llm_helper:
    def __init__(self, args, q, out_q):
        self.args = args
        self.out_q = out_q
        self.q = q

    def loop(self):
        if self.q.empty() == True:
            return
        
        text = self.q.get()
        response = ""
        if self.args.llm == 'gemini':
            response = gemini.fetch_response(text)
            self.out_q.put(response)
        else:
            print("No llm model selected bhai >_<")

        print(f"llm >> {response}")
        
