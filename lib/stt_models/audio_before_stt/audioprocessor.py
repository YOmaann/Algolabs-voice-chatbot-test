import numpy as np
from scipy.signal import butter, sosfilt_zi, sosfilt
from pyrnnoise.pyrnnoise import RNNoise
import torch
from torchaudio.transforms import Resample

class AudioProcessor:
    def __init__(self, sample_rate=16000, filtering=True, noise_suppression=True):

        self.sample_rate = sample_rate

        self.filtering = filtering
        self.noise_suppression = noise_suppression

        #initialising filtering, noise reduction parameters
        self.bp_sos = None
        self.zi = None

        if self.filtering:
            self.initialise_filter_state()

        if self.noise_suppression:
            self.denoiser = RNNoise(sample_rate=16000)
            self.upsampler = Resample(orig_freq=16000, new_freq=48000, dtype=torch.float32)
            self.downsampler = Resample(orig_freq=48000, new_freq=16000, dtype=torch.float32)
        

    def initialise_filter_state(self, pass_type = "highpass", cutoff = 80):
        '''Digital Filter
        Args:
            pass_type: "highpass", "lowpass", "bandpass"
            cutoff = scalar for highpass or lowpass, array/list for bandpass'''
        # by default, a very light highpass filter : to remove AC hums and such
        nyquist = self.sample_rate/2
        self.bp_sos = butter(2, np.array(cutoff)/nyquist, btype=pass_type, output="sos")
        self.zi = sosfilt_zi(self.bp_sos)

    def _filtering(self, audio_chunk):
        # sosfilt returns float64
        audio_chunk, self.zi = sosfilt(self.bp_sos, audio_chunk, zi=self.zi)
        return audio_chunk.astype(np.float32)
    
    def _noise_suppression(self, audio_chunk):
        '''
        Args:
            audio_chunk: np.ndarray of np.float32'''
        # checked rnnoise at 16khz : BAD
        # rnnoise wants 48kHz, float32
        # resample audio at 48kHz: convert to tensor and use torchaudio for resampling, for better RNNoise working, can cause overhead
        # audio: np_float32 -> tensor (+unsqueeze) -> upsample -> np_float32 -> np_int16 -> process -> np_float32 -> tensor -> downsample -> np_float32

        try: 
            audio_tensor = torch.from_numpy(audio_chunk)

            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0) #torch resampling expects shape [channels, samples]

            #checking if working
            print(f"Input audio_chunk dtype: {audio_chunk.dtype}, one thing: {audio_chunk[0].dtype}")
            print(f"Audio tensor dtype: {audio_tensor.dtype}")
    
            audio_upsampled = self.upsampler(audio_tensor)

            if audio_upsampled.dim() > 1:
                back_to_np = audio_upsampled.squeeze(0).numpy()  # remove channel dim
            else:
                back_to_np = audio_upsampled.numpy()

            audio_clamped = np.clip(back_to_np, -1.0, 1.0)
            audio_int16 = (audio_clamped * 32767).astype(np.int16)

            temp = self.denoiser.process_chunk(audio_int16)

            probs = [] #probability of speech in that chunk of audio_chunk (rnnoise breaks audio_chunk into chunks of 480 frames each)
            processed_audio = [] 
            for prob, frame in temp:
                prob = prob[0][0]
                probs.append(prob)
                frame_1d = frame.flatten() if frame.ndim > 1 else frame
                processed_audio.extend(frame_1d)

            processed_audio = np.array(processed_audio)
            processed_audio_float32 = processed_audio.astype(np.float32) / 32767.0
            processed_tensor = torch.from_numpy(processed_audio_float32)
            if processed_tensor.dim() == 1:
                processed_tensor = processed_tensor.unsqueeze(0)

            audio_downsampled = self.downsampler(processed_tensor)

            if audio_downsampled.dim() > 1:
                final_audio = audio_downsampled.squeeze(0).numpy()
            else:
                final_audio = audio_downsampled.numpy()
            return final_audio, probs   

        except Exception as e:
            print(f"Encountered exception: {e}\n Not suppressing noise.")
    
    def process_chunk(self, audio_chunk, probability=False):
        if self.filtering:
            audio_chunk = self._filtering(audio_chunk)
        if self.noise_suppression:
            audio_chunk, probs = self._noise_suppression(audio_chunk)
            # see if/how to use probs later
            if probability:
                return audio_chunk, probs
        return audio_chunk