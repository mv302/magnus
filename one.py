import streamlit as st
import speech_recognition as sr
from transformers import pipeline
from audiorecorder import audiorecorder

r = sr.Recognizer()

def recognize(audio):
    st.write("Recording stopped")
    with sr.WavFile('audio.wav') as source:     
        a = r.record(source)
        try:
            text = r.recognize_google(a)
            st.write("You said:", text)
        except sr.UnknownValueError:
            text = ""
            st.write("Sorry, I could not understand audio.")
            exit()

    pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")
    result = pipe(text)[0]["label"]
    st.warning("Emotion detected")
    st.subheader(f"You spoke in :blue[{result}] tone")
    # st.write("Accuracy: ", result[0]["score"] )
    
def main():
    st.title("Magnus")
    audio = audiorecorder("Start recording", "Stop recording")
    st.write(':green[Press press start recording and stop to analyse it\'s emotion]')
    if len(audio) > 0:
        st.audio(audio.export().read())  
        audio.export("audio.wav", format="wav")
        st.write(f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds")
        recognize(sr.AudioFile("audio.wav"))

if __name__ == "__main__":
    main()
