import numpy as np

def audio_data_to_float32(audio_data):
    raw_data = audio_data.get_raw_data()
    sample_rate = audio_data.sample_rate
    sample_width = audio_data.sample_width

    if sample_width == 2:
        waveform = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        waveform = np.frombuffer(raw_data, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    return waveform, sample_rate
