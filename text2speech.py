from gtts import gTTS
import os
tts = gTTS(text='Hello world!', lang='fr')
tts.save("good.mp3")
#os.system("mpg321 good.mp3")
os.system("afplay good.mp3")
