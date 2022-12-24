import cv2
import speech_recognition as sr

# obtain audio from the microphone
r = sr.Recognizer()


# recognize speech using Google Speech Recognition
try:
	# for testing purposes, we're just using the default API key
	# to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
	# instead of `r.recognize_google(audio)`
	# print("Google Speech Recognition thinks you said in English: -  " + r.recognize_google(audio, language = "en-US"))
	while True:
		key = cv2.waitKey(1)
		if key == ord("s"):
			with sr.Microphone() as source:
				print("Say something!")
				audio = r.listen(source)

		elif key == ord("q"):
			print("Google Speech Recognition thinks you said in Uzbeistan: -  " + r.recognize_google(audio, language="uz-UZ"))
			break


	# print("Google Speech Recognition thinks you said in Uzbeistan: -  " + r.recognize_google(audio, language = "uz-UZ"))
except sr.UnknownValueError:
	print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
	print("Could not lts from Google Speech Recognition service; {0}".format(e))