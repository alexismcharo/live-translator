import os, json, uuid, tempfile, subprocess, torch, whisper, uvicorn, openai
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv

# === ENV & MODELS ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = openai.AsyncOpenAI(api_key=api_key)

# Load Whisper
model = whisper.load_model("large-v3")

# Load DeepSeek R1
device = "cuda" if torch.cuda.is_available() else "cpu"
deepseek_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-translate", trust_remote_code=True)
deepseek_model = AutoModelForSeq2SeqLM.from_pretrained("deepseek-ai/deepseek-translate", trust_remote_code=True).to(device)

# === FastAPI Setup ===
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

transcript_history = []  # [(segment_id, original_text)]

# === Warmup ===
try:
    subprocess.run([
        "ffmpeg", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
        "-t", "0.5", "-ar", "16000", "-ac", "1",
        "-y", "/tmp/warm.wav"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except:
    pass

try:
    model.transcribe("/tmp/warm.wav", language="en", fp16=True)
except:
    pass

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join("frontend", "index.html"))

# === WebSocket Route ===
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

                    # Transcribe or Translate
                    if source_lang == "Japanese":
                        result = model.transcribe(
                            wav.name,
                            fp16=True,
                            task="translate",
                            language="ja",
                            temperature=0.0,
                            beam_size=1,
                            condition_on_previous_text=True,
                            hallucination_silence_threshold=0.2,
                            no_speech_threshold=0.3,
                            compression_ratio_threshold=2.4,
                            logprob_threshold=-1.0
                        )
                        text = result["text"].strip()
                        print("🎌 Whisper-translated from Japanese:", text)
                    else:
                        result = model.transcribe(
                            wav.name,
                            fp16=True,
                            task="transcribe",
                            language="en",
                            temperature=0.0,
                            beam_size=1,
                            condition_on_previous_text=True,
                            hallucination_silence_threshold=0.2,
                            no_speech_threshold=0.3,
                            compression_ratio_threshold=2.4,
                            logprob_threshold=-1.0
                        )
                        text = result["text"].strip()
                        print("🇬🇧 Transcribed English:", text)

                    if not text:
                        continue

                    # Skip thank-you filler
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

                    # Hallucination check
                    if await hallucination_check(text):
                        print("🧠 GPT flagged as hallucination:", text)
                        continue

                    segment_id = str(uuid.uuid4())
                    transcript_history.append((segment_id, text))

                    translated = await translate_text(text, source_lang, target_lang)
                    await websocket.send_text(f"[DONE]{json.dumps({'id': segment_id, 'text': translated})}")

                    if len(transcript_history) >= 2:
                        prev, curr = transcript_history[-2][1], transcript_history[-1][1]
                        improved = await translate_text((prev, curr), source_lang, target_lang, mode="context")
                        await websocket.send_text(f"[UPDATE]{json.dumps({'id': transcript_history[-2][0], 'text': improved})}")

    except Exception as e:
        print("❌ WebSocket error:", e)
        await websocket.close()

# === Translation Pipeline: GPT + DeepSeek ===
async def translate_text(text, source_lang, target_lang, mode="default"):
    try:
        # Step 1: Contextual refinement via GPT-4o
        if mode == "context":
            previous, current = text
            prompt = (
                f"You are a strict, literal translation engine. Refine the translation of the previous sentence based on new context.\n\n"
                f"Previous: {previous}\n"
                f"Current: {current}\n\n"
                f"Rules:\n"
                f"- Do NOT repeat previous sentences unless their meaning changes.\n"
                f"- Merge or rephrase ONLY if new information adds clarity.\n"
                f"- Return the improved translation of the 'Previous' sentence only.\n"
                f"- Do NOT repeat phrases that were already translated unless absolutely necessary.\n"
                f"- If a sentence is identical or near-identical to the previous one, translate it only once.\n"
            )
        else:
            prompt = (
                f"You are a strict, literal translation engine. Translate the sentence below from {source_lang} to {target_lang}.\n\n"
                f"Rules:\n"
                f"- Return only the translated sentence. No commentary, no meta info.\n"
                f"- Never say 'already in English/Japanese'. If it's already translated, return it as-is.\n"
                f"- No thank yous, greetings, or filler.\n\n"
                f"Sentence: {text}"
            )

        gpt_response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a strict and literal translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=100
        )
        refined_text = gpt_response.choices[0].message.content.strip()

        # Step 2: Final polish via DeepSeek
        prefix = "Translate English to Japanese: " if target_lang == "Japanese" else "Translate Japanese to English: "
        input_text = prefix + refined_text
        inputs = deepseek_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
        output_ids = deepseek_model.generate(**inputs, max_new_tokens=256)
        output_text = deepseek_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        return output_text.strip()

    except Exception as e:
        print("❌ Translation error:", e)
        return text

# === GPT-based Hallucination Filter ===
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

# === App Runner ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
