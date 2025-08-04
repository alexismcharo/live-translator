import tempfile, subprocess, uvicorn, openai, whisper, os, json, uuid
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = openai.AsyncOpenAI(api_key=api_key)

try:
    subprocess.run([
        "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
        "-t", "0.5", "-ar", "16000", "-ac", "1",
        "-y", "/tmp/warm.wav"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except:
    pass

model = whisper.load_model("large-v3")

try:
    model.transcribe("/tmp/warm.wav", language="en", fp16=True)
except:
    pass

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

transcript_history = []  # [(segment_id, original_text)]

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join("frontend", "index.html"))

# hallucination filter via GPT-4o
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
            "ONLY return:\nYES — if it's a known filler phrase\nNO — if it's natural speech, even if vague or uncertain\n\n"
            f"Sentence:\n{text}"
        )

        result = await client.chat.completions.create(
            model="gpt-4o",
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
            spoof_language = "ja"  # the hack: lie to Whisper about input
        elif direction == "ja-en":
            spoof_language = "en"
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
                            "-af", "silenceremove=1:0:-40dB",
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
                        beam_size=1,
                        condition_on_previous_text=True,
                        hallucination_silence_threshold=0.2,
                        no_speech_threshold=0.3,
                        language=spoof_language,
                        compression_ratio_threshold=2.4,
                        logprob_threshold=-1.0,
                        task="translate"
                    )

                    text = result["text"].strip()
                    print("📝 Transcribed:", text)
                    if not text:
                        continue

                    # filter thank-you phrases
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

                    # hallucination check
                    if await hallucination_check(text):
                        print("🧠 GPT flagged as hallucination:", text)
                        continue

                    # assign ID and send result
                    segment_id = str(uuid.uuid4())
                    transcript_history.append((segment_id, text))
                    await websocket.send_text(f"[DONE]{json.dumps({'id': segment_id, 'text': text})}")

    except Exception as e:
        print("❌ WebSocket error:", e)
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
