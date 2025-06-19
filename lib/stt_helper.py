import time
import speech_recognition as sr
from queue import Queue

# class wrapper to allow extra parameters in the callback functipn.
class speech_callback:
    def __init__(self, _args, queue):
        self.args = _args
        self.queue = queue

        # Custom recognizer initializers
        print('Loading STT model..')
        if _args.stt == 'faster_whisper':
            from lib.stt_models.faster_whisper import get_recognizer
            self.recognize_faster_whisper = get_recognizer()

        print('Loaded STT model :)')    
            
    
    def _cb(self, r, audio):
        print("Listened a phrase ;)))")
        try:
            txt = ""
            if self.args.stt == 'google':
                txt = r.recognize_google(audio) # add custom key using key parameter
                # print(txt)
                self.queue.put(txt)
            elif self.args.stt == 'sphinx':
                txt = r.recognize_sphinx(audio)
                self.queue.put(txt)
            elif self.args.stt == 'whisper':
                txt = r.recognize_whisper(audio)
                self.queue.put(txt)
            elif self.args.stt == 'faster_whisper':
                txt = self.recognize_faster_whisper(audio)
                self.queue.put(txt)
            elif self.args.stt == 'openai':
                txt = r.recognize_openai(audio)
                self.queue.put(txt)
            elif self.args.stt == 'groq':
                txt = r.recognize_groq(audio)
                self.queue.put(txt)
            elif self.args.stt == 'vosk':
                txt = r.recognize_vosk(audio)
                self.queue.put(txt)
            else:
                print('No model selected bhai ))')

            print(f"stt>> {txt}")

        except sr.UnknownValueError:
            print("Model cannot recogonize your voice :(")
        except sr.RequestError as e:
            print(f"Model gave an error: {e}")
        
            
        
class speech_helper:
    def __init__(self, args, queue):
        self.args = args
        self.queue = queue
        self.r = sr.Recognizer()
        self.m = sr.Microphone()

        with self.m as source:
            self.r.adjust_for_ambient_noise(source)

    
    def loop(self):
        stop_listening = self.r.listen_in_background(self.m, speech_callback(self.args, self.queue)._cb)

        # for _ in range(50): time.sleep(0.1) # time.sleep blocks the execution of the main thread. :-)

        return stop_listening
        #stop_listening(wait_for_stop=False)
