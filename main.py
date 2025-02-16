import os
import uuid
import asyncio
import argparse
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from google import genai
from google.cloud import speech_v1p1beta1 as speech
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from browser_use import Agent
import asyncio
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
import json
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS"
)

app = Flask(__name__) 

client = genai.Client(api_key=os.getenv("gemini_api_key"))


def record_audio(
    sample_rate: int, duration: float, filename: str = "recording.wav"
) -> str:
    """Records audio from the microphone until the duration is reached."""
    print(f"Recording audio for {duration} seconds...")
    recording = sd.rec(
        int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype="float32"
    )
    sd.wait()
    print(f"Recording finished. Saving to {filename}...")
    sf.write(filename, recording, sample_rate)
    print(f"Audio saved to {filename}")
    return filename


def transcribe_audio(audio_file_path: str) -> str:
    """Transcribes the given audio file using Google Cloud Speech API."""
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
        for alternative in result.alternatives:
            transcription += alternative.transcript
    return transcription


async def web_agent(answer):
    answer_json = json.loads(answer)
    load_dotenv()

    api_key = os.getenv("gemini_api_key")
    task_str = f"""
    The user has described the following issue: "{answer_json['user_issue']}".
    A potential solution proposed is: "{answer_json['solution']}".
    Considering the details above, please analyze and identify a specific product that best addresses the user's requirements.
    Provide a brief explanation on why this product is a suitable choice.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp", api_key=SecretStr(api_key)
    )
    agent = Agent(task=task_str, llm=llm)
    result = await agent.run()
    print(result)


class AnswerSchema(BaseModel):
    user_issue: str
    solution: str
    product: str


def generate_answer(prompt: str) -> str:
    """Generates an answer using the Gemini model given a prompt with a structured JSON response."""

    sys_instruct = """
    Based on a given user's prompt, you are to write an explanation for what:
      - the user's issue is,
      - how to potentially solve it,
      - and suggest the type of product needed to help them.
    
    Please respond using the following JSON schema:
      {
        "user_issue": "User's issue",
        "solution": "Potential solution",
        "product": "Product suggestion"
      }
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config={
            "system_instruction": sys_instruct,
            "response_mime_type": "application/json",
            "response_schema": AnswerSchema,
        },
    )
    return response.text


async def main():
    parser = argparse.ArgumentParser(
        description="Record, transcribe, and generate an answer."
    )
    parser.add_argument(
        "--duration", type=float, default=5.0, help="Recording duration in seconds"
    )
    args = parser.parse_args()

    sample_rate = 16000
    audio_dir = "audio_files"
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
    unique_filename = os.path.join(audio_dir, f"{uuid.uuid4()}.wav")

    loop = asyncio.get_running_loop()

    try:
        recorded_file = await loop.run_in_executor(
            None, record_audio, sample_rate, args.duration, unique_filename
        )
        transcript = await loop.run_in_executor(None, transcribe_audio, recorded_file)
        answer = await loop.run_in_executor(None, generate_answer, transcript)

        print("\n--- Results ---")
        print("Transcript:")
        print(transcript)
        print("\nAnswer:")
        print(answer)

        await web_agent(answer)
    except Exception as e:
        print(f"Error processing audio: {e}")
    finally:
        if os.path.exists(unique_filename):
            os.remove(unique_filename)


@app.route('/transcribe', methods=['POST'])
async def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    unique_filename = f"{uuid.uuid4()}.wav"
    audio_dir = "audio_files"
    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
    audio_file_path = os.path.join(audio_dir, unique_filename)

    audio_file.save(audio_file_path)

    try:
        loop = asyncio.get_running_loop()
        transcript = await loop.run_in_executor(None, transcribe_audio, audio_file_path)
        answer = await loop.run_in_executor(None, generate_answer, transcript)

        web_agent_result = await web_agent(answer)  # Await web_agent

        return jsonify({
            'transcription': transcript,
            'answer': answer,
            'web_agent_result': web_agent_result
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)


if __name__ == "__main__":
    app.run(debug=True)  # Development server only


