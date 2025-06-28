import pyaudio
import numpy as np
import threading
import queue
import time
from audioprocessor import AudioProcessor
import soundfile as sf

class RealTimeAudio:
    def __init__(self, chunk_size=8192, sample_rate=16000, channels=1, format=pyaudio.paFloat32, preprocess=True):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate # might have to upsample for usecase: 8khz used often in phonic 
        self.channels = channels
        self.format = format

        self.p = pyaudio.PyAudio()
        self.audio_queue = queue.Queue()
        self.audio_list = []

        self.recording = False
        self.recording_thread = None

        self.stream = None

        if preprocess:
            self.audio_processor = AudioProcessor()  
        else:
            self.audio_processor = AudioProcessor(filtering=False, noise_suppression=False)

    def start_recording(self, input_device=1):
        self.recording = True

        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=input_device,
            frames_per_buffer=self.chunk_size)
        
        self.recording_thread = threading.Thread(target=self._record_audio)
        self.recording_thread.daemon = True
        self.recording_thread.start()
        

    def _record_audio(self):

        print("Recording begins:")
        while self.recording:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.float32)
                
                # audio processing
                audio_data = self.audio_processor.process_chunk(audio_data)

                self.audio_queue.put(audio_data)
                self.audio_list.extend(list(audio_data))

            except Exception as e:
                print(f"Error during recording: {e}")
                break

    def stop_recording(self):
        self.recording = False

        if self.recording_thread:
            self.recording_thread.join(timeout=1) #use is_alive to check if thread is still alive or join has been called

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        self.p.terminate()
        print("Recording stopped.")

    def get_audio_data(self):
        return self.audio_queue

    def get_audio_as_array(self):
        audio_array = np.array(self.audio_list)
        return audio_array
    
    def save_as_wav(self, filename):
        audio_array = self.get_audio_as_array()
        audio_array = audio_array.flatten()
        sf.write(f"{filename}.wav", audio_array, 16000)

def main(record_time = 8, save = True, filename = None, preprocess=True):
    recorder = RealTimeAudio(channels=1, preprocess=preprocess)
    #start_time = time.time()
    recorder.start_recording(input_device=1)
    time.sleep(record_time)
    recorder.stop_recording()
    if save and filename:
        recorder.save_as_wav(filename)
    elif save:
        from uuid import uuid4
        recorder.save_as_wav(f"output{uuid4()}")

if __name__ == "__main__":
    main(record_time=15, preprocess=True, save=True, filename="output_sabconversionskardiya")