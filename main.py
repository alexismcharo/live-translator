import tempfile, subprocess, uvicorn, openai, whisper, os, json, uuid
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = openai.AsyncOpenAI(api_key=api_key)

model = whisper.load_model("medium")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

transcript_history = []  # [(segment_id, original_text)]

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join("frontend", "index.html"))

async def hallucination_check(text):
    try:
        prompt = (
            "You are a strict hallucination filter. Your task is to detect whether a sentence sounds like a generic, fabricated, or AI-generated filler phrase.\n\n"
            "Examples:\n"
            "- 'Thanks for watching'\n"
            "- 'Don't forget to subscribe'\n"
            "- 'Click the bell icon'\n"
            "- 'See you in the next video'\n"
            "- 'Like and share'\n\n"
            "If the sentence contains any such content, return only: YES\nOtherwise: NO\n\n"
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
    except:
        return False

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔌 WebSocket connected")

    try:
        settings = await websocket.receive_text()
        config = json.loads(settings)
        direction = config.get("direction")
        if direction == "en-ja":
            source_lang, target_lang = "English", "Japanese"
        elif direction == "ja-en":
            source_lang, target_lang = "Japanese", "English"
        else:
            await websocket.close()
            return

        while True:
            msg = await websocket.receive()
            if "bytes" not in msg:
                continue

            audio = msg["bytes"]
            if not audio:
                continue

            with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as raw:
                raw.write(audio)
                raw.flush()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav:
                    try:
                        subprocess.run([
                            "ffmpeg", "-y",
                            "-i", raw.name,
                            "-af", "silenceremove=1:0:-50dB",
                            "-ar", "16000",
                            "-ac", "1",
                            wav.name
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        check=True
                        )
                    except:
                        continue

                    result = model.transcribe(
                        wav.name,
                        fp16=True,
                        temperature=0.0,
                        beam_size=5,
                        condition_on_previous_text=False,
                        hallucination_silence_threshold=0.2,
                        no_speech_threshold=0.3,
                        language="en" if source_lang == "English" else "ja",
                        compression_ratio_threshold=2.4,
                        logprob_threshold=-1.0
                    )

                    text = result["text"].strip()
                    print("📝 Transcribed:", text)
                    if not text:
                        continue

                    # skip generic thank-you/subscribe phrases (hardcoded filter)
                    text_lower = text.lower()
                    if (
                        "thank you" in text_lower
                        or "thanks" in text_lower
                        or "ありがとう" in text
                        or "ありがとうございます" in text
                        or "ありがと" in text
                    ):
                        print("🚫 Skipping thank-you/ありがとう phrase:", text)
                        continue

                    # optional GPT hallucination filter
                    if await hallucination_check(text):
                        print("🧠 GPT flagged as hallucination:", text)
                        continue

                    # store and translate
                    segment_id = str(uuid.uuid4())
                    transcript_history.append((segment_id, text))

                    translated = await translate_text(text, source_lang, target_lang)
                    await websocket.send_text(f"[DONE]{json.dumps({'id': segment_id, 'text': translated})}")

                    # context-aware update for previous segment
                    if len(transcript_history) >= 2:
                        context_segs = transcript_history[-2:]
                        context_text = " ".join(s[1] for s in context_segs)
                        improved = await translate_text(context_text, source_lang, target_lang)
                        await websocket.send_text(f"[UPDATE]{json.dumps({'id': context_segs[0][0], 'text': improved})}")

    except Exception as e:
        print("❌ WebSocket error:", str(e))
        await websocket.close()

async def translate_text(text, source_lang, target_lang):
    prompt = (
        f"You are a strict, literal translation engine. Translate the sentence below from {source_lang} to {target_lang}.\n\n"
        f"Rules:\n"
        f"- Return only the translated sentence. No commentary, no meta info.\n"
        f"- Never say 'already in English/Japanese'. If it's already translated, return it as-is.\n"
        f"- No thank yous, greetings, or filler.\n\n"
        f"Sentence: {text}"
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a strict and literal translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ Translation error:", e)
        return text

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
