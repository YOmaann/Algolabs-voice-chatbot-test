import pyaudio
import numpy as np
import threading
import queue
import time
from scipy.signal import butter, sosfilt_zi, sosfilt
import noisereduce as nr #can also explore rnnoise 
#import silero_vad
import wave

class AudioProcessor:
    def __init__(self, sample_rate=16000, filtering=False, noise_suppression=True, normalisation=True):
        self.sample_rate = sample_rate
        self.filtering = filtering # not defaulting to True bcz its not recommneded for whister/faster-whisper: has internal preprocessing

        self.noise_suppression = noise_suppression
        self.normalisation = normalisation

        #initialising bandpass, noise reduction and normalisation parameters
        self.cutoff = [300,3400]
        self.pass_type = "band"
        self.bp_sos = None
        self.zi = None

        self.nr_prop_decrease = 0.95 #how much % to decrease noise
        self.noise_profile = None
        self.noise_profile_ready = False

        
        self.target_rms = 0.1
        self.norm_alpha = 0.95
        self.running_rms = 0.1
        self.max_gain = 0.3


    def initialise_noise_profile(self, noise_profile):
        self.noise_profile = noise_profile
        self.noise_profile_ready = True

    def initialise_filter_state(self, pass_type = "bandpass", cutoff = [300, 3400]):
        '''Digital Filter
        Args:
            pass_type: "highpass", "lowpass", "bandpass"
            cutoff = scalar for highpass or lowpass, array/list for bandpass'''
        nyquist = self.sample_rate/2
        self.bp_sos = butter(2, np.array(cutoff)/nyquist, btype=pass_type, output="sos")
        self.zi = sosfilt_zi(self.bp_sos)

    def _filtering(self, audio_chunk):
        audio_chunk, self.zi = sosfilt(self.bp_sos, audio_chunk, zi=self.zi)
        return audio_chunk

    def _normalisation(self, audio_chunk):
        '''Using Adaptive RMS-based Normalization with Exponential Smoothing''' 
        current_rms = np.sqrt(np.mean(audio_chunk**2)) #change ow its just 1 audio chunk
        if current_rms > 1e-6:
            self.running_rms = self.norm_alpha * self.running_rms + (1 - self.norm_alpha) * current_rms
            if self.running_rms > 1e-6:
                gain = self.target_rms / self.running_rms
                gain = np.clip(gain, 0.05, self.max_gain) #limiting gain range
                audio_chunk = audio_chunk * gain
    
        return audio_chunk
        
    def _noise_suppression(self, audio_chunk):            
        if self.noise_profile_ready:
            try:
                noise_suppressed_chunk = nr.reduce_noise(
                    y=audio_chunk,
                    sr=self.sample_rate,
                    y_noise=self.noise_profile,
                    prop_decrease=self.nr_prop_decrease
                )
                audio_chunk = noise_suppressed_chunk

            except Exception as e:
                print(f"Error during noise reduction: {e}")
            
            finally:
                return audio_chunk

    def process_chunk(self, audio_chunk):
        if self.filtering:
            audio_chunk = self._filtering(audio_chunk)
        if self.noise_suppression:
            audio_chunk = self._noise_suppression(audio_chunk)
        if self.normalisation:
            audio_chunk = self._normalisation(audio_chunk)
        return audio_chunk    

class RealTimeAudio:
    def __init__(self, chunk_size=8192, sample_rate=16000, channels=1, format=pyaudio.paFloat32):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate # might have to upsample for usecase: 8khz used often in phonic 
        self.channels = channels
        self.format = format

        self.p = pyaudio.PyAudio()
        self.audio_queue = queue.Queue()

        self.recording = False
        self.recording_thread = None

        self.stream = None

        self.noise_profile = []
        self.noise_profiling_time = 3 #seconds
        self.noise_profile_ready = False
        self.audio_processor = AudioProcessor()  

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
        if not self.noise_profile_ready and self.audio_processor.noise_suppression:
            try:
                print("Calibrating noise")
                start = time.time()
                while time.time()-start < self.noise_profiling_time:
                    # collecting inital noise profile. say something like "please wait for calibration."
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    audio_data = np.frombuffer(data, dtype=np.float32)
                    self.noise_profile.append(audio_data)
                self.noise_profile = np.concatenate(self.noise_profile)
                self.noise_profile_ready = True 
                self.audio_processor.initialise_noise_profile(self.noise_profile)
                print("Calibration over. Beginning recording.")
            except Exception as e:
                print(f"Error during noise calibration: {e}")

        while self.recording:
            try:
                if self.audio_processor.filtering:
                    self.audio_processor.initialise_filter_state()
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.float32)
                # audio processing
                audio_data = self.audio_processor.process_chunk(audio_data)
                self.audio_queue.put(audio_data)

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




        
