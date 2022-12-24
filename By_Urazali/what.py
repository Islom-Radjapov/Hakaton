import speech_recognition as sr

# obtain audio from the microphone
r = sr.Recognizer()
with sr.Microphone() as source:
	print("Say something!")
	audio = r.listen(source)

# recognize speech using Google Speech Recognition
try:
	print("Google Speech Recognition thinks you said in Uzbeistan: -  " + r.recognize_google(audio, language="uz-UZ"))
except sr.UnknownValueError:
	print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
	print("Could not lts from Google Speech Recognition service; {0}".format(e))