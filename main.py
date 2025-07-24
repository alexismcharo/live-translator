import tempfile, subprocess, uvicorn, openai, whisper, os, json
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = openai.AsyncOpenAI(api_key=api_key)

model = whisper.load_model("large-v3")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

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
        print("📥 Received config:", settings)
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
                            "-af", "silenceremove=1:0:-50dB",  # Trim silence
                            "-ar", "16000",
                            "-ac", "1",
                            wav.name
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        check=True
                        )

                        # skip short audio (< 0.5s)
                        probe = subprocess.run(
                            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                             "-of", "default=noprint_wrappers=1:nokey=1", wav.name],
                            capture_output=True, text=True
                        )
                        duration = float(probe.stdout.strip())
                        if duration < 0.5:
                            print("⏭️ Skipping short audio chunk")
                            continue

                    except subprocess.CalledProcessError as e:
                        print("❌ FFmpeg error: could not convert audio chunk")
                        print("🔎 FFmpeg stderr:", e.stderr.decode())
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

                    # Filter known hallucinated phrases
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

                    # optional GPT filter
                    if await hallucination_check(text):
                        print("🧠 GPT flagged as hallucination:", text)
                        continue

                    await stream_translate_with_gpt(websocket, text, source_lang, target_lang)

    except Exception as e:
        print("❌ WebSocket error:", str(e))
        await websocket.close()

async def stream_translate_with_gpt(websocket, text, source_lang, target_lang):
    try:
        prompt = (
            f"You are a strict, literal translation engine. Translate the sentence below from {source_lang} to {target_lang}.\n\n"
            f"Rules:\n"
            f"- Return only the translated sentence. No commentary, no meta info.\n"
            f"- Never say 'already in English/Japanese'. If it's already translated, return it as-is.\n"
            f"- No thank yous, greetings, or filler.\n\n"
            f"Sentence: {text}"
        )

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

    except Exception as e:
        print("❌ Translation error:", e)
        try:
            await websocket.send_text("[DONE]")
        except:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
