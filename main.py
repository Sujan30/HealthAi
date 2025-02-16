import io
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import time
import uuid
import threading
from dotenv import load_dotenv
from google import genai
from flask import Flask, request, jsonify

from google.cloud import speech_v1p1beta1 as speech



app = Flask(__name__)  # Add Flask app initialization

load_dotenv()

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

client = genai.Client(api_key=os.getenv("gemini_api_key"))

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Explain how AI works",
)




def record_audio(sample_rate, duration, filename="recording.wav"):
    """Records audio until duration is reached or Ctrl+C is pressed."""

    print(f"Recording audio for {duration} seconds (or until Ctrl+C)...")

    recording = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  
    print(f"Audio recording finished. Saving to {filename}...")
    sf.write(filename, recording, sample_rate)
    print(f"Audio saved to {filename}")
    return filename

#processing the prompt by the user

def transcribe_audio(audio_file_path):
    client = speech.SpeechClient()

    with open(audio_file_path, "rb") as audio:
        content = audio.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, 
        sample_rate_hertz=16000, 
        language_code="en-US",  #
        enable_automatic_punctuation=True, 
    )

    response = client.recognize(config=config, audio=audio)

    transcription = ""
    for result in response.results:
        for alternative in result.alternatives:
            transcription += alternative.transcript
    return transcription

#using the prompt, look for keywords that the user is interested in that is health realated


#currently uses  to generate the answer

def generate_answer(prompt):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return response.text



@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    audio_file_path = 'uploaded_audio.wav'
    audio_file.save(audio_file_path)
    transcription = transcribe_audio(audio_file_path)

    try:
        full_transcription = transcribe_audio(audio_file_path)

        #the transcription is then used as the prompt for the language model
        
        return jsonify({'transcription': full_transcription})
    
    except Exception as e:
        return jsonify({'error': 'Error transcribing audio'}), 500
    
def main():
    
    sample_rate = 16000
    duration = float(input("Enter recording duration (in seconds): "))
    
    # Create unique filename with proper extension
    unique_filename = str(uuid.uuid4()) + ".wav"
    file_path = os.path.join("audio_files", unique_filename)
    
    try:
        if not os.path.exists("audio_files"):
            os.makedirs("audio_files")
            
        recorded_file = record_audio(sample_rate, duration, file_path)
        transcript = transcribe_audio(recorded_file)
        print(f"Full Transcript of prompt: {transcript}")
        print(f"Answer: {generate_answer(transcript)}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
    
    



if __name__ == '__main__':
    main()

      
    

