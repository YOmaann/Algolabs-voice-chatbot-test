import speech_recognition as sr
from faster_whisper import WhisperModel
import soundfile as sf
import torch
import io
from utils.audio import audio_data_to_float32



def get_recognizer(model_size = 'large-v3', language = 'hi'):
    model = WhisperModel(model_size, device="cuda" if torch.cuda.is_available() else "cpu")
    def recognize_whisper(audio):
        waveform, sample_rate = audio_data_to_float32(audio)

        _buffer = io.BytesIO()
        sf.write(_buffer, waveform, sample_rate, format="WAV")
        _buffer.seek(0) 

        segments, _ = model.transcribe(_buffer, language, beam_size=5)
        
        transcription = " ".join([seg.text for seg in segments])
        return transcription
    return recognize_whisper
