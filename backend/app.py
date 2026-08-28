"""FastAPI application entry point."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FRONTEND_DIR, LANGUAGES, MODEL_PATH  # noqa: E402
from backend.services.gesture_service import GestureRecognitionService  # noqa: E402
from utils.language_maps import GESTURE_TRANSLATIONS, list_gestures  # noqa: E402
from utils.tts_utils import get_browser_tts_code, synthesize_speech  # noqa: E402

from ml_model.classifier import GestureClassifier  # noqa: E402

app = FastAPI(title="AI Sign Language Translator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

shared_classifier = GestureClassifier(MODEL_PATH)

app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists() and shared_classifier.is_trained,
        "languages": LANGUAGES,
    }


@app.get("/api/gestures")
async def gestures():
    return {
        "gestures": list_gestures(),
        "translations": GESTURE_TRANSLATIONS,
    }


@app.post("/api/tts")
async def text_to_speech(payload: dict):
    text = payload.get("text", "").strip()
    language = payload.get("language", "en")
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    audio = synthesize_speech(text, language)
    if audio is None:
        return JSONResponse(
            {
                "fallback": "browser",
                "tts_code": get_browser_tts_code(language),
                "text": text,
            }
        )

    from fastapi.responses import Response

    return Response(content=audio, media_type="audio/mpeg")


@app.websocket("/ws/recognize")
async def recognize_ws(websocket: WebSocket):
    await websocket.accept()
    service = GestureRecognitionService(classifier=shared_classifier)
    language = "en"

    try:
        while True:
            message = await websocket.receive_json()
            msg_type = message.get("type")

            if msg_type == "config":
                language = message.get("language", "en")
                continue

            if msg_type == "reset":
                service.reset()
                await websocket.send_json({"type": "reset", "translated_text": ""})
                continue

            if msg_type == "clear":
                service.clear_text()
                await websocket.send_json(
                    {"type": "clear", "translated_text": ""}
                )
                continue

            if msg_type == "frame":
                frame_b64 = message.get("data", "")
                frame = service.decode_frame(frame_b64)
                if frame is None:
                    await websocket.send_json({"type": "error", "message": "Bad frame"})
                    continue

                result = service.process_frame(frame, language)
                await websocket.send_json({"type": "result", **result})

    except WebSocketDisconnect:
        pass
    finally:
        service.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
