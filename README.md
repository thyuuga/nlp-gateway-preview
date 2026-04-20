# nlp-gateway

A multi-capability NLP microservice gateway built with FastAPI, serving as the unified inference backend for a conversational AI system.

## Architecture

```
app.py                          # FastAPI entry, router registration, startup banner
routers/                        # HTTP layer — request/response models, validation
  ├── judge.py                  # Promise acceptance judgment
  ├── time_parse.py             # Temporal expression parsing
  ├── features_promise.py       # Promise candidate feature extraction
  ├── hard_write.py             # Explicit memory-write command parsing
  ├── embed_qualify.py          # Embedding quality gate
  ├── embed_encode.py           # Text → vector encoding
  ├── emotion.py                # Single-sentence emotion classification
  └── tts.py                    # Text-to-speech synthesis
services/                       # Business logic — models, NLP pipelines, clients
  ├── judge_logic.py            # Regex-based acceptance/rejection detection
  ├── duckling_client.py        # Facebook Duckling temporal parser client
  ├── promise_features.py       # Multi-signal promise detection (time + action + intent)
  ├── hard_write_logic.py       # Profile field extraction via HanLP + BERT classifier
  ├── profile_parse.py          # User profile parsing (HanLP tokenization + V-O extraction)
  ├── embedding_service.py      # Sentence-Transformers lazy-loaded encoder
  ├── emotion_service.py        # MacBERT 6-class emotion classifier
  ├── tts_service.py            # OpenAI TTS wrapper with stage-direction removal
  ├── lang_guess.py             # Language detection heuristic
  ├── nlp.py                    # spaCy NLP utilities
  └── message_filter/           # Strategy pattern — extensible per-language filtering
        ├── base.py             # Abstract FilterStrategy + FilterResult
        └── chinese.py          # Chinese filter: HanLP SRL/POS scoring + fast-path regex
models/                         # Pre-trained model artifacts
  ├── emotion_model/            # Fine-tuned MacBERT for 6-class emotion
  └── profile_model/            # Fine-tuned MacBERT for profile field classification
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | FastAPI + Uvicorn |
| Chinese NLP | HanLP (tokenization, POS, NER, SRL) |
| Emotion Classification | MacBERT fine-tuned (6 classes: neutral / happy / sad / angry / proud / shy) |
| Profile Classification | MacBERT fine-tuned (multi-label profile field detection) |
| Embeddings | Sentence-Transformers |
| Temporal Parsing | Facebook Duckling |
| TTS | OpenAI gpt-4o-mini-tts |
| General NLP | spaCy |

## Design Highlights

- **Router-Service Separation** — Clean HTTP layer (routers) decoupled from business logic (services). Each router only handles request validation and response formatting.

- **Lazy Loading** — All heavy models (HanLP, MacBERT, Sentence-Transformers) are loaded on first use, not at startup. This keeps cold start fast and memory efficient when not all endpoints are needed.

- **Strategy Pattern** — Message filtering uses an abstract `FilterStrategy` base class with language-specific implementations (`ChineseFilterStrategy`). New languages can be added by implementing the interface.

- **Trace ID Propagation** — Every request carries a `x-trace-id` header, logged consistently across all services for end-to-end debugging.

- **Graceful Fallback** — Missing model weights (`.safetensors`) don't crash the server. Affected endpoints return 503 while all other endpoints remain operational.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/judge/accept` | POST | Promise acceptance judgment |
| `/features/time/parse` | POST | Temporal expression parsing (via Duckling) |
| `/features/promise/candidate` | POST | Promise candidate feature extraction |
| `/hard_write/judge` | POST | Explicit write-command parsing |
| `/embed/qualify` | POST | Embedding quality gate (SRL + POS scoring) |
| `/embed/encode` | POST | Text → vector encoding |
| `/emotion/analyze` | POST | Single-sentence emotion classification |
| `/tts/speak` | POST | Text-to-speech (returns MP3 audio stream) |

## Pre-trained Models

Model weight files (`.safetensors`) are excluded from git. After cloning, copy them manually:

```bash
# emotion_model — 6-class emotion classifier (MacBERT fine-tuned)
cp /path/to/emotion_model/model.safetensors models/emotion_model/

# profile_model — profile field classifier (MacBERT fine-tuned)
cp /path/to/profile_model/model.safetensors models/profile_model/
```

Config and tokenizer files are tracked in git.

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
cp .env.example .env  # then fill in OPENAI_API_KEY, etc.

# 4. Run
uvicorn app:app --host 127.0.0.1 --port 8123 --log-level info
```

## Deployment

This service runs on a **Raspberry Pi 4B (ARM64, CPU-only)** as part of a larger conversational AI system.

**Runtime dependencies:**

| Service | Purpose | Default Address |
|---------|---------|-----------------|
| nlp-gateway (this) | NLP inference | `http://127.0.0.1:8123` |
| Facebook Duckling | Temporal expression parsing | `http://127.0.0.1:8001` |

Duckling is deployed as a separate process on the same host. It provides structured time/date extraction (e.g. "下周三下午三点" → ISO 8601) used by the `/features/time/parse` and `/features/promise/candidate` endpoints.

**Resource considerations:**

- All ML models run on CPU — PyTorch is installed with `--index-url https://download.pytorch.org/whl/cpu` to minimize binary size
- Embedding model (`BAAI/bge-small-zh-v1.5`, ~95MB) chosen for low memory footprint
- Lazy loading ensures only requested models are loaded into memory (~2-3GB peak when all models are active)

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | For TTS | OpenAI API key for text-to-speech |
| `DUCKLING_URL` | Optional | Duckling server URL (default: `http://127.0.0.1:8001/parse`) |
| `LOG_LEVEL` | Optional | Logging level (default: `INFO`) |
