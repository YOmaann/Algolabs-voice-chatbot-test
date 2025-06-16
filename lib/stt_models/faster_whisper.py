import speech_recognition as sr
from faster_whisper import WhisperModel
import soundfile as sf
import torch
import io



def get_recognizer(model_size = 'large-v3', language = 'in'):
    model = WhisperModel(model_size, device="cuda" if torch.cuda.is_available() else "cpu")
    def recognize_whisper(audio):

        audio_data = audio.get_wav_data()
        audio_file = io.BytesIO(audio_data)
        audio_array, samplerate = sf.read(audio_file)

        segments, _ = model.transcribe(audio_array, language, beam_size=5)
        
        transcription = " ".join([seg.text for seg in segments])
        return transcription
    return recognize_whisper
