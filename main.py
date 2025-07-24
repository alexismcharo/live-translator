import tempfile, subprocess, uvicorn, openai, whisper, os, re, json
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY is missing from .env")

client = openai.AsyncOpenAI(api_key=api_key)
model = whisper.load_model("large")  

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join("frontend", "index.html"))

# Heuristic check to determine whether a sentence is likely complete
def is_complete_sentence(text, lang):
    text = text.strip()
    if lang == "Japanese":
        patterns = [r"です$", r"ます$", r"でした$", r"だ$", r"よ$", r"ね$", r"んです$", r"でしょう$", r"か[。？\?]?$", r"[。！？?!]$"]
        return any(re.search(p, text) for p in patterns) or len(text.split()) > 6
    return bool(re.search(r"[.!?]$", text))

# Uses GPT to classify whether a sentence sounds like generic filler
async def hallucination_check(text):
    try:
        prompt = (
            "You are a strict hallucination filter. Your task is to detect whether a sentence sounds like a generic, fabricated, or AI-generated filler phrase.\n\n"
            "Examples of such hallucinations include phrases like:\n"
            "- 'Thanks for watching'\n"
            "- 'Don't forget to subscribe'\n"
            "- 'Click the bell icon'\n"
            "- 'See you in the next video'\n"
            "- 'Like and share'\n\n"
            "If the sentence contains any such content, return only:\n"
            "YES\n\n"
            "If the sentence is meaningful and not hallucinated, return only:\n"
            "NO\n\n"
            f"Sentence:\n{text}"
        )

        result = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You judge if the sentence is hallucinated filler. Only reply YES or NO."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=1
        )
        return result.choices[0].message.content.strip().upper() == "YES"
    except Exception:
        return False  # fallback: allow through if GPT call fails

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    sentence_buffer = ""
    source_lang = target_lang = last_sent = None

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

                    # Filter based on confidence and silence
                    if result.get("no_speech_prob", 0) > 0.3 and result.get("avg_logprob", -0.6) < -1.4:
                        continue

                    text = result["text"].strip()
                    if not text:
                        continue

                    sentence_buffer += " " + text
                    sentence_buffer = sentence_buffer.strip()

                    if not is_complete_sentence(sentence_buffer, source_lang):
                        continue
                    if await hallucination_check(sentence_buffer):
                        sentence_buffer = ""
                        continue

                    await stream_translate_with_gpt(websocket, sentence_buffer, source_lang, target_lang)
                    last_sent = sentence_buffer
                    sentence_buffer = ""

    except Exception:
        await websocket.close()

# Sends translation stream using GPT
async def stream_translate_with_gpt(websocket, text, source_lang, target_lang):
    prompt = (
        f"You are a strict, literal translation engine. Translate the sentence below from {source_lang} to {target_lang}.\n\n"
        f"Rules:\n"
        f"- Return only the translated sentence. No commentary, no meta info.\n"
        f"- Never say 'already in English/Japanese'. If it's already translated, return it as-is.\n"
        f"- No thank yous, greetings, or filler.\n\n"
        f"Sentence: {text}"
    )

    try:
        stream = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a strict and literal translator."},
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
