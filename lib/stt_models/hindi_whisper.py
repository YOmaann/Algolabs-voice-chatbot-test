import torch
from transformers import pipeline, logging
from datasets import load_from_disk
import numpy as np
from utils.audio import audio_data_to_float32


def get_recognizer():
    logging.set_verbosity_error() 
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="./models/whisper_hindi",
        chunk_length_s=30,
        device=device
    )
    
    def recognize_whisper(audio):
        waveform, sample_rate = audio_data_to_float32(audio)
        
        prediction = asr_pipe({
            'name' : 'test.wav',
            'array' : waveform,
            'sampling_rate' : sample_rate,
            'shape' : (waveform.size,)
        })

        transcription = prediction['text']

        # print('done')
        
        return transcription
    return recognize_whisper
