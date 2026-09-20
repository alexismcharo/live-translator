# Socious Live Translator

A browser-based **English ↔ Japanese live translation prototype** built during my software engineering internship at Socious.

The application captures microphone audio in the browser, sends short audio chunks to a FastAPI backend over WebSockets, transcribes the audio using OpenAI Whisper, and translates the resulting text with the OpenAI API. Translated captions are then returned to the browser in near real time.

## Problem

Socious works across English- and Japanese-speaking teams. This prototype was designed to make live conversations easier to follow by providing translated captions during meetings.

The main engineering challenge was balancing:

- translation latency;
- speech-recognition accuracy;
- sufficient conversational context; and
- suppression of repeated or hallucinated ASR output.

## How it works

1. The user selects **English → Japanese** or **Japanese → English**.
2. The browser records short microphone chunks using the MediaRecorder API.
3. Audio is sent to the FastAPI backend through a WebSocket connection.
4. FFmpeg converts incoming audio to 16 kHz mono WAV.
5. Whisper `large-v3` transcribes the audio.
6. Low-value or likely hallucinated output is filtered.
7. A small amount of recent context is supplied to the translation model.
8. The translated caption is sent back to the browser and displayed.

## Key features

- English ↔ Japanese live translation
- Browser microphone capture
- WebSocket-based audio transport
- Whisper `large-v3` transcription
- OpenAI-powered translation
- FFmpeg audio conversion
- Conversational context handling
- Duplicate and redundant-output suppression
- Filtering of common ASR hallucinations and low-value interjections
- Responsive browser interface
- Docker and NVIDIA GPU support
- Whisper and FFmpeg pre-warming to reduce first-request latency

## Technologies

### Backend

- Python
- FastAPI
- WebSockets
- OpenAI API
- OpenAI Whisper
- FFmpeg
- Uvicorn

### Frontend

- HTML
- JavaScript
- Tailwind CSS
- MediaRecorder API
- WebSocket API

### Infrastructure

- Docker
- Docker Compose
- NVIDIA CUDA runtime

## My contribution

I developed and iterated on the prototype, including the browser recording workflow, FastAPI/WebSocket backend, Whisper transcription pipeline, OpenAI translation logic, filtering and context handling.

A significant part of the work involved reducing perceived latency while preserving transcription quality. I experimented with different audio chunk sizes and Whisper decoding settings, reduced unnecessary model calls, pre-warmed Whisper and FFmpeg, and refined the translation prompt and context handling.

The prototype was used internally to help English- and Japanese-speaking people within the organisation follow each other during meetings.

## Project structure

```text
.
├── frontend/
│   └── index.html       # Browser UI, microphone capture and WebSocket client
├── main.py              # FastAPI server, ASR, translation and filtering
├── requirements.txt     # Python dependencies
├── Dockerfile           # CUDA-enabled application image
└── docker-compose.yml   # GPU-enabled local container setup
