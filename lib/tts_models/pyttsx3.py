import pyttsx3


def text_to_speech(text):
    # romanized hindi text to speech
    engine = pyttsx3.init()

    # engine.setProperty('voice', 'hindi')
    engine.setProperty('volume', 0.9)

    # engine.say(text)
    engine.say(text) # need romanised hindi
    engine.runAndWait()

