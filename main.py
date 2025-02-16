import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
from google.cloud import speech_v1p1beta1 as speech
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
import requests

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for the frontend

client = genai.Client(api_key=os.getenv("gemini_api_key"))

def transcribe_audio(audio_file_path: str) -> str:
    speech_client = speech.SpeechClient()
    with open(audio_file_path, "rb") as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
        enable_automatic_punctuation=True,
    )
    response = speech_client.recognize(config=config, audio=audio)
    transcription = ""
    for result in response.results:
        transcription += result.alternatives[0].transcript
    return transcription

def generate_answer(prompt: str) -> str:
    sys_instruct = """
    Based on a given user's prompt, provide a detailed 
    """
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config={
            "system_instruction": sys_instruct,
            "response_mime_type": "application/json",
        },
    )
    return response.text

@app.route("/transcribe", methods=["POST"])
def transcribe_endpoint():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    audio_file = request.files["audio"]
    filename = f"temp_{uuid.uuid4()}.wav"
    audio_file.save(filename)
    try:
        transcription = transcribe_audio(filename)
        return jsonify({"transcription": transcription})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filename):
            os.remove(filename)

@app.route("/generate-answer", methods=["POST"])
def generate_answer_endpoint():
    try:
        data = request.get_json()
        if not data or "prompt" not in data:
            return jsonify({"error": "No prompt provided"}), 400

        prompt = data["prompt"]
        video_id = "67b20478b0350420b9874478"  # example ID
        url = "https://api.twelvelabs.io/v1.3/generate"
        payload = {
            "prompt": prompt,
            "temperature": 0.2,
            "stream": True,
            "video_id": video_id,
        }
        headers = {
            "accept": "application/json",
            "x-api-key": os.getenv("elevenlabs_api_key"),
            "Content-Type": "application/json",
        }
        # Make the external request
        response = requests.post(url, json=payload, headers=headers)

        # Return the raw text as "response"
        return jsonify({"response": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
