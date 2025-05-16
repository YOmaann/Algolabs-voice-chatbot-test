# from ai4bharat.transliteration import XlitEngine # could use this too lazy
import lib.tts_models.pyttsx3 as pyttx
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate

def is_hindi(text):
    return any('\u0900' <= ch <= '\u097F' for ch in text)

class tts_helper:
    def __init__(self, args, out_q):
        self.args = args
        self.out_q = out_q

    def loop(self):
        if self.out_q.empty() == True:
            return
        
        text = self.out_q.get()

        if self.args.tts == 'pyttsx3':
            if is_hindi(text):
                text = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS) # ITRANS
            # need conversion from romanised -> devanagri (logic of the program)
            print(f"tts>> {text}")
            pyttx.text_to_speech(text)
            print("Said :)))))")
        
