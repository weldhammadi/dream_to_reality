# backend.py

import base64
import os
from mistralai import Mistral
from dotenv import load_dotenv
from groq import Groq
import json
import math
import tempfile
import requests
from datetime import datetime
import sqlite3
import uuid

load_dotenv()

def init_database():
    """Initialize SQLite database for history."""
    conn = sqlite3.connect('audio_to_image_history.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS generations (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            transcribed_text TEXT,
            emotion_analysis TEXT,
            generated_prompt TEXT,
            image_path TEXT,
            content_analysis TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def save_to_history(transcribed_text, emotion_analysis, generated_prompt, image_path, content_analysis):
    """Save generation data to history database."""
    conn = sqlite3.connect('audio_to_image_history.db')
    cursor = conn.cursor()
    
    generation_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    cursor.execute('''
        INSERT INTO generations (id, timestamp, transcribed_text, emotion_analysis, generated_prompt, image_path, content_analysis)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (generation_id, timestamp, transcribed_text, json.dumps(emotion_analysis), generated_prompt, image_path, json.dumps(content_analysis)))
    
    conn.commit()
    conn.close()
    
    return generation_id

def get_history(limit=10):
    """Get generation history from database."""
    conn = sqlite3.connect('audio_to_image_history.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM generations 
        ORDER BY timestamp DESC 
        LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            'id': row[0],
            'timestamp': row[1],
            'transcribed_text': row[2],
            'emotion_analysis': json.loads(row[3]) if row[3] else {},
            'generated_prompt': row[4],
            'image_path': row[5],
            'content_analysis': json.loads(row[6]) if row[6] else {}
        })
    
    return history

def delete_from_history(generation_id):
    """Delete a generation from history."""
    conn = sqlite3.connect('audio_to_image_history.db')
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM generations WHERE id = ?', (generation_id,))
    
    conn.commit()
    conn.close()

def read_file(file_path):
    with open(file_path, "r") as file:
        return file.read()

def softmax(predictions):
    output = {}
    for sentiment, predicted_value in predictions.items():
        output[sentiment] = math.exp(predicted_value*10) / sum([math.exp(value*10) for value in predictions.values()])
    return output

def speach_to_text(audio_path, language="fr"):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    with open(audio_path, "rb") as file:

        transcription = client.audio.transcriptions.create(
            file=file, # Required audio file
            model="whisper-large-v3-turbo", # Required model to use for transcription
            prompt="Extrait le text de l'audio de la manière la plus factuelle possible",  # Optional
            response_format="verbose_json",  # Optional
            timestamp_granularities = ["word", "segment"], # Optional (must set response_format to "json" to use and can specify "word", "segment" (default), or both)
            language="fr",  # Optional
            temperature=0.0  # Optional
        )

        return transcription.text

def generate_image_prompt(transcribed_text):
    """Generate an image prompt from transcribed text using Mistral AI."""
    import time
    
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    role_content = read_file("./role.txt")
    
    # Try with primary model first, fallback to smaller model
    models_to_try = ["mistral-large-latest", "mistral-small-latest"]
    
    for model in models_to_try:
        # Retry logic for rate limiting
        max_retries = 1
        for attempt in range(max_retries):
            try:
                chat_response = client.chat.complete(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": role_content
                        },
                        {
                            "role": "user",
                            "content": f"Génère un prompt d'image détaillé et créatif basé sur ce texte transcrit realiste 4k 9/16 like its a shot from a scene : {transcribed_text}"
                        }
                    ]
                )
                return chat_response.choices[0].message.content
                
            except Exception as e:
                if "429" in str(e):
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 2  # Exponential backoff: 10s, 20s, 40s
                        print(f"Rate limit hit with {model}, waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    else:
                        print(f"Rate limit exceeded for {model}, trying next model...")
                        break  # Try next model
                else:
                    raise e
    
    # If all models fail, raise the last error
    raise Exception("All Mistral AI models are currently rate limited. Please try again later.")

def generate_image_with_clipdrop(prompt):
    """Generate an image using ClipDrop API."""
    api_key = os.environ.get("CLIPDROP_API_KEY")
    if not api_key:
        raise ValueError("CLIPDROP_API_KEY not found in environment variables")
    
    url = 'https://clipdrop-api.co/text-to-image/v1'
    
    files = {
        'prompt': (None, prompt, 'text/plain')
    }
    
    headers = {
        'x-api-key': api_key
    }
    
    response = requests.post(url, files=files, headers=headers)
    
    if response.ok:
        return response.content
    else:
        raise Exception(f"ClipDrop API error: {response.status_code} - {response.text}")

def text_analysis(text):

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    chat_response = client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {
            "role": "system",
            "content": read_file(text_file_path="./context_analysis.txt")
            },
            {
            "role": "user",
            "content": f"Analyse le texte ci-dessous (ta réponse doit être dans le format JSON) : {text}",
            },
        ],
        response_format={"type": "json_object",}
    )

    predictions = json.loads(chat_response.choices[0].message.content)
    
    return softmax(predictions)

def encode_image(image_data):
    """Encode an image file-like object or path to base64."""
    try:
        if hasattr(image_data, "read"):  # BytesIO from streamlit
            return base64.b64encode(image_data.read()).decode('utf-8')
        else:  # Path to file
            with open(image_data, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error: {e}")
        return None

def describe_image(image_input):
    """Takes either a file path or BytesIO image and returns Mistral output."""
    base64_image = encode_image(image_input)
    if not base64_image:
        return "Erreur lors de l'encodage de l'image."

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    messages = [
        {
            "role": "system",
            "content": read_file("./context.txt")
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": read_file("prompt.txt")
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            ]
        }
    ]

    chat_response = client.chat.complete(
        model="pixtral-12b-2409",
        messages=messages
    )

    return chat_response.choices[0].message.content

def transcribe_audio(audio_input):
    """Transcribe audio using Groq."""
    try:
        # Handle different audio input types
        if hasattr(audio_input, "read"):  # BytesIO from streamlit
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_input.read())
                temp_file_path = temp_file.name
        else:  # Path to file
            temp_file_path = audio_input
            
        text = speach_to_text(temp_file_path, language="fr")
        
        # Clean up temporary files
        if hasattr(audio_input, "read"):
            os.unlink(temp_file_path)
                
        return text
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def analyze_audio_text(transcribed_text):
    """Analyze transcribed text using Mistral AI."""
    if not transcribed_text:
        return "Erreur lors de la transcription de l'audio."
    
    return text_analysis(transcribed_text)

def analyze_content_emotions(transcribed_text):
    """Analyze emotions in transcribed text using Mistral AI."""
    import time
    
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    
    # Try with primary model first, fallback to smaller model
    models_to_try = ["mistral-large-latest", "mistral-small-latest"]
    
    for model in models_to_try:
        # Retry logic for rate limiting
        max_retries = 3
        for attempt in range(max_retries):
            try:
                chat_response = client.chat.complete(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": read_file("./context_analysis.txt")
                        },
                        {
                            "role": "user",
                            "content": f"Analyse le texte ci-dessous (ta réponse doit être dans le format JSON) : {transcribed_text}",
                        },
                    ],
                    response_format={"type": "json_object",}
                )
                
                predictions = json.loads(chat_response.choices[0].message.content)
                return softmax(predictions)
                
            except Exception as e:
                if "429" in str(e):
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 10  # Exponential backoff: 10s, 20s, 40s
                        print(f"Rate limit hit with {model}, waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    else:
                        print(f"Rate limit exceeded for {model}, trying next model...")
                        break  # Try next model
                else:
                    raise e
    
    # If all models fail, raise the last error
    raise Exception("All Mistral AI models are currently rate limited. Please try again later.")

def analyze_content_themes(transcribed_text):
    """Analyze content themes and topics using Mistral AI."""
    import time
    
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    
    # Try with primary model first, fallback to smaller model
    models_to_try = ["mistral-large-latest", "mistral-small-latest"]
    
    for model in models_to_try:
        # Retry logic for rate limiting
        max_retries = 3
        for attempt in range(max_retries):
            try:
                chat_response = client.chat.complete(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": """Tu es un assistant d'analyse de contenu. Tu dois analyser le texte et identifier les thèmes principaux.
                            Renvoie STRICTEMENT un objet JSON avec les champs suivants (valeurs entre 0 et 1):
                            - nature: contenu lié à la nature, paysages
                            - urbain: contenu lié à la ville, architecture
                            - personnes: contenu lié aux personnes, portraits
                            - objets: contenu lié aux objets, nature morte
                            - abstrait: contenu abstrait, conceptuel
                            - action: contenu dynamique, mouvement
                            - calme: contenu paisible, statique"""
                        },
                        {
                            "role": "user",
                            "content": f"Analyse les thèmes de ce texte : {transcribed_text}",
                        },
                    ],
                    response_format={"type": "json_object",}
                )
                
                predictions = json.loads(chat_response.choices[0].message.content)
                return softmax(predictions)
                
            except Exception as e:
                if "429" in str(e):
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) * 10  # Exponential backoff: 10s, 20s, 40s
                        print(f"Rate limit hit with {model}, waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                    else:
                        print(f"Rate limit exceeded for {model}, trying next model...")
                        break  # Try next model
                else:
                    raise e
    
    # If all models fail, raise the last error
    raise Exception("All Mistral AI models are currently rate limited. Please try again later.")

if __name__ == "__main__":
    init_database()  # Initialize the database
    image_description = describe_image(r"D:\school\HETIC\PYTHON\rève\design_my_haircut-main\WIN_20250703_12_52_11_Pro.jpg")
    print(image_description)