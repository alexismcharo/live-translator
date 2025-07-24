import tempfile
import subprocess
import uvicorn
import openai
import whisper
import os
import re
import json
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY is missing from .env")

# OpenAI client
client = openai.OpenAI(api_key=api_key)

# Load Whisper model
model = whisper.load_model("medium")

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# GPT translation
def translate_with_gpt(text, source_lang, target_lang):
    prompt = (
        f"You are a translator. Translate the following sentence from {source_lang} to {target_lang}. "
        f"Do not explain or interpret. Return only the translated sentence.\n\n"
        f"Sentence: {text}"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a precise, literal translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ GPT translation error:", e)
        return "[Translation error]"

# hybrid sentence completeness check
def is_complete_sentence(text, lang):
    text = text.strip()
    if lang == "Japanese":
        patterns = [
            r"です$", r"ます$", r"でした$", r"だ$", r"よ$", r"ね$", r"んです$", r"でしょう$", 
            r"か[。？\?]?$", r"[。！？?!]$"
        ]
        return any(re.search(p, text) for p in patterns) or len(text.split()) > 6
    else:
        return bool(re.search(r"[.!?]$", text))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    sentence_buffer = ""
    source_lang = None
    target_lang = None

    try:
        # Receive initial settings
        settings = await websocket.receive_text()
        try:
            config = json.loads(settings)
            direction = config.get("direction")
            if direction == "en-ja":
                source_lang, target_lang = "English", "Japanese"
            elif direction == "ja-en":
                source_lang, target_lang = "Japanese", "English"
            else:
                await websocket.send_text("[Invalid translation direction]")
                await websocket.close()
                return
        except json.JSONDecodeError:
            await websocket.send_text("[Failed to parse translation settings]")
            await websocket.close()
            return

        while True:
            print("🟡 Waiting for audio chunk...")
            audio_bytes = await websocket.receive_bytes()
            print(f"🟢 Received chunk: {len(audio_bytes)} bytes")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as input_file:
                input_file.write(audio_bytes)
                input_file.flush()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as output_wav:
                    try:
                        subprocess.run(
                            ["ffmpeg", "-y", "-i", input_file.name, "-ar", "16000", "-ac", "1", output_wav.name],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                            check=True
                        )

                        response = model.transcribe(output_wav.name, fp16=True)
                        chunk_text = response["text"].strip()
                        print(f"📝 Transcript: '{chunk_text}'")

                        if not chunk_text:
                            print("⚠️ Skipping short/empty chunk")
                            continue

                        sentence_buffer += " " + chunk_text
                        sentence_buffer = sentence_buffer.strip()

                        if is_complete_sentence(sentence_buffer, source_lang):
                            print(f"✅ Sentence complete: {sentence_buffer}")
                            translated = translate_with_gpt(sentence_buffer, source_lang, target_lang)
                            print("🌍 Translated:", translated)
                            await websocket.send_text(translated)
                            sentence_buffer = ""
                        else:
                            print("⏳ Waiting for sentence to complete...")

                    except subprocess.CalledProcessError as e:
                        print("❌ FFmpeg error:\n", e.stderr.decode())
                    except Exception as err:
                        print("❌ Processing error:\n", err)

    except Exception as e:
        print("🚫 WebSocket error:", e)
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
