# backend_with_hallucination_filters.py
import tempfile, subprocess, uvicorn, openai, whisper, os, re, json, difflib
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY is missing from .env")

client = openai.AsyncOpenAI(api_key=api_key)
model = whisper.load_model("medium")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join("frontend", "index.html"))

def is_complete_sentence(text, lang):
    text = text.strip()
    if lang == "Japanese":
        patterns = [r"です$", r"ます$", r"でした$", r"だ$", r"よ$", r"ね$", r"んです$", r"でしょう$", r"か[。？\?]?$", r"[。！？?!]$"]
        return any(re.search(p, text) for p in patterns) or len(text.split()) > 6
    return bool(re.search(r"[.!?]$", text))

def is_too_similar(a, b, threshold=0.92):
    return difflib.SequenceMatcher(None, a, b).ratio() > threshold

async def hallucination_check(text, source_lang):
    try:
        prompt = f"""You're a hallucination detector. If the following sentence sounds like a fabricated or generic phrase often produced by AI (e.g., “Thank you for watching”, “Subscribe”, etc.), return only: YES.
Otherwise, return only: NO.
Sentence: {text}"""
        result = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You judge whether a sentence is a hallucinated or common filler phrase. Answer strictly YES or NO."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1
        )
        reply = result.choices[0].message.content.strip().upper()
        return reply == "YES"
    except Exception:
        return False  # fallback to not blocking
        

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    sentence_buffer = ""
    source_lang, target_lang, last_sent = None, None, None

    try:
        settings = await websocket.receive_text()
        config = json.loads(settings)
        if config.get("direction") == "en-ja":
            source_lang, target_lang = "English", "Japanese"
        elif config.get("direction") == "ja-en":
            source_lang, target_lang = "Japanese", "English"
        else:
            await websocket.close()
            return

        while True:
            msg = await websocket.receive()
            if "bytes" not in msg:
                continue

            audio = msg["bytes"]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as raw:
                raw.write(audio)
                raw.flush()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav:
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", raw.name, "-ar", "16000", "-ac", "1", wav.name],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                    )

                    result = model.transcribe(wav.name, fp16=True, word_timestamps=False)
                    if result.get("no_speech_prob", 0) > 0.2 or result.get("avg_logprob", -0.6) < -1.0:
                        continue

                    text = result["text"].strip()
                    if not text:
                        continue

                    sentence_buffer += " " + text
                    sentence_buffer = sentence_buffer.strip()

                    if not is_complete_sentence(sentence_buffer, source_lang):
                        continue

                    if last_sent and is_too_similar(sentence_buffer, last_sent):
                        sentence_buffer = ""
                        continue

                    if await hallucination_check(sentence_buffer, source_lang):
                        sentence_buffer = ""
                        continue

                    # Translate
                    await stream_translate_with_gpt(websocket, sentence_buffer, source_lang, target_lang)
                    last_sent = sentence_buffer
                    sentence_buffer = ""

    except Exception:
        await websocket.close()

async def stream_translate_with_gpt(websocket, text, source_lang, target_lang):
    prompt = (
    f"You are a strict, literal translation engine. Translate the following sentence from {source_lang} to {target_lang}.\n\n"
    f"Your rules:\n"
    f"- Only return the translated sentence — no commentary, no explanations, no repetition of the original.\n"
    f"- Do NOT add greetings, closings, thank yous, or YouTube-style phrases.\n"
    f"- Preserve tone and formality; do not embellish or simplify.\n"
    f"- If the input is already in the target language, do NOT say so — just return the sentence unchanged.\n"
    f"- Do NOT say 'already in English' or similar — silently return the original.\n\n"
    f"Sentence: {text}")


    try:
        stream = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a precise, literal translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=150,
            stream=True
        )

        full = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full += delta
                await websocket.send_text(f"[STREAM]{full}")
        await websocket.send_text(f"[DONE]{full}")

    except Exception:
        await websocket.send_text("[DONE]")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
