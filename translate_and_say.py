from translate import Translator
from gtts import gTTS
import os

def ts(text):
    translator = Translator(to_lang="ru", from_lang="en")
    translation = translator.translate(text)
    tts = gTTS(text=translation, lang='ru')
    tts.save("tmp.mp3")
    os.system("afplay tmp.mp3")

#ts("Stop coronavirus, one more time!")