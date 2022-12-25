import tkinter as tk
from tkinter import *
import pyttsx3

speech = pyttsx3.init()
root = Tk()
root.title("Text to Speech")
root.geometry("500x200")

txt = StringVar()

def mySpeak():
    speech.say(txt.get())
    speech.runAndWait()
    speech.stop()

Label(root, text="Matnni nutuqga o'zgartirish!", font=("Helvetica 20 bold"), fg="Green").pack(pady=10)

frame1 = Frame(root)
Label(frame1, text="Nimadir yozing: ", font=("Helvetica 15 bold")).pack(side=LEFT)
text = Entry(frame1, textvariable=txt, font=("Helvetica 15"))
text.pack()
frame1.pack(pady=5)

Button(root, text="Gapiring", font=("Helvetica 15 bold"), command=mySpeak).pack()

root.mainloop()