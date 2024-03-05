import streamlit as st
import speech_recognition as sr
from transformers import pipeline

r = sr.Recognizer()

def recognize(audio):
    st.write("Recording stopped")
    try:
        text = r.recognize_google(audio)
        st.write("You said:", text)
    except sr.UnknownValueError:
        text = ""
        st.write("Sorry, I could not understand audio.")

    pipe = pipeline("text-classification")
    st.write(pipe(text))
    
def main():
    st.title("Magnus")
    with sr.Microphone() as source:  
        record = st.button("Record", type="primary")
        if record:
            st.write('Recording...')
            audio = r.listen(source)
            recognize(audio)

if __name__ == "__main__":
    main()