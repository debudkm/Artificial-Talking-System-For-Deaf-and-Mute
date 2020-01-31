from gtts import gTTS
import os
mytext = 'Welcome to python gtts'
language = 'en'
myobj = gTTS(text=mytext, lang=language, slow=False)
myobj.save("welcome.mp3")

# Playing the converted file
os.system("mpg321 welcome.mp3")