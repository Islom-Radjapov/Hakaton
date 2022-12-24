import speech_recognition as sr

# obtain audio from the microphone
def name():
    r = sr.Recognizer()
    
    with sr.Microphone() as source:
	    print("Say something!")
	    audio = r.listen(source)

# recognize speech using Google Speech Recognition
	# for testing purposes, we're just using the default API key
	# to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
	# instead of `r.recognize_google(audio)`
        try:
	        print("Google Speech Recognition thinks you said in English: -  " + r.recognize_google(audio, language = "en-US"))
	        print("Google Speech Recognition thinks you said in Turkish: -  " + r.recognize_google(audio, language = "uz-UZ"))


        except sr.UnknownValueError:
	        print("Google Speech Recognition could not understand audio")


        except sr.RequestError as e:
	        print("Could not request results from Google Speech Recognition service; {0}".format(e))

if __name__ == "__main__":
    main()