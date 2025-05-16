import argparse
from lib.stt_helper import speech_helper
from lib.llm_helper import llm_helper
from lib.tts_helper import tts_helper
import time
from queue import Queue

def main_block(args):
    print('Starting..:0')
    q = Queue() # stt output queue
    out_q = Queue() # llm output queue
    # Initialize class objects.
    stt = speech_helper(args, q)
    llm = llm_helper(args, q, out_q)
    tts = tts_helper(args, out_q)
    # stt_helper needs to be executed once. explain later.
    stt_stopper = stt.loop()
    # Event loop
    try:
        while True:
            llm.loop()
            tts.loop()
    except KeyboardInterrupt:
        print('Byi Bye :)')
        
    
    

    stt_stopper(wait_for_stop=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Model-test',
                    description='Program to test different voice and text models using SpeechRecognition Library',
                    epilog='Made without love </>')

    parser.add_argument('-stt', '--stt', type=str, help='Choose speech-to-text engine')
    parser.add_argument('-tts', '--tts', type=str, help='Choose text-to-speech engine')
    parser.add_argument('-llm', '--llm', type=str, help='Choose language model')
    args = parser.parse_args()
    print(args)
    main_block(args)
