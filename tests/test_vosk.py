#!/usr/bin/env python3

import wave
import sys
import time

from vosk import Model, KaldiRecognizer, SetLogLevel

# You can set log level to -1 to disable debug messages
SetLogLevel(0)

wf = wave.open(sys.argv[1], "rb")
if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
    print("Audio file must be WAV format mono PCM.")
    sys.exit(1)

wf2 = wave.open(sys.argv[2], "rb")
if wf2.getnchannels() != 1 or wf2.getsampwidth() != 2 or wf2.getcomptype() != "NONE":
    print("Audio file must be WAV format mono PCM.")
    sys.exit(1)



# model = Model(lang="en-us")

# You can also init model by name or with a folder path
# model = Model(model_name="vosk-model-en-us-0.21")
model = Model("../models/vosk-model-hi-0.22")

rec = KaldiRecognizer(model, wf.getframerate())
rec.SetWords(True)
rec.SetPartialWords(True)

startT = time.time()

while True:
    data = wf.readframes(12000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        print(rec.Result())
    else:
        print(rec.PartialResult())

print(rec.FinalResult())
print(time.time() - startT)

startT = time.time()

while True:
    data = wf2.readframes(12000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        print(rec.Result())
    else:
        print(rec.PartialResult())


print(rec.FinalResult())
print(time.time() - startT)
