import argparse
from lib.stt_helper import speech_helper
from lib.llm_helper import llm_helper
from lib.tts_helper import tts_helper
import time
from queue import Queue
from threading import Thread, Event

def main_block(args):
    print('Starting..:0')
    q = Queue() # stt output queue
    out_q = Queue() # llm output queue

    # Define a signal variable
    stop_e = Event()
    # Initialize class objects.
    stt = speech_helper(args, q)
    llm = llm_helper(args, q, out_q, stop_e)
    tts = tts_helper(args, out_q, )
    # stt_helper needs to be executed once. explain later.
    stt_stopper = stt.loop()

    llm_thread = Thread(target = llm.loop)
    llm_thread.start() # Start thread
    
    # Event loop
    try:
        while True:
            # Can implement a safeword feature - which makes the tts stop by saying specific keywords. Requires creating a seperate thread for tts loop and killing that. Very ugly :((.
            tts.loop()
    except KeyboardInterrupt:
        print('Byi Bye :)')
    finally:
        stop_e.set() # Stop llm
        stt_stopper(wait_for_stop=False)
        llm_thread.join()
        
    time.sleep(0.2) # let everthing cool down ~_~
    

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
