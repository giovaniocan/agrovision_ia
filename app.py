import json
import os
import sqlite3
import threading
import time
from typing import List

import cv2
import requests
import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from ultralytics import YOLO

from services.scraping import scrape_agro_news


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configuracoes
DB_PATH = "agrovision.db"
YOLO_MODEL_NAME = "yolo11n.pt"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# Estado do chat
chat_history: List["Message"] = []
chat_history_lock = threading.Lock()

# Estado da visibilidade
latest_frame = None
latest_frame_lock = threading.Lock()
visibility_state = {
    "status": "aguardando analise",
    "details": "A camera ainda nao gerou dados suficientes.",
    "updated_at": None,
}
visibility_lock = threading.Lock()

# Inicializa YOLO
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = YOLO(YOLO_MODEL_NAME)
model.to(device)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[Message] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    history: List[Message]


def message_to_dict(message: Message) -> dict:
    if hasattr(message, "model_dump"):
        return message.model_dump()
    return message.dict()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            label TEXT,
            confidence REAL,
            image_path TEXT
        )
        """
    )
    conn.close()


init_db()


def get_last_event():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, event_time, label, confidence, image_path
        FROM events
        ORDER BY id DESC
        LIMIT 1
        """
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_chat_history() -> List[Message]:
    with chat_history_lock:
        return [Message(role=item.role, content=item.content) for item in chat_history]


def set_chat_history(history: List[Message]) -> None:
    with chat_history_lock:
        chat_history.clear()
        chat_history.extend(Message(role=item.role, content=item.content) for item in history)


def append_chat_exchange(user_message: str, assistant_message: str, base_history: List[Message]) -> List[Message]:
    updated_history = [
        *base_history,
        Message(role="user", content=user_message),
        Message(role="assistant", content=assistant_message),
    ]
    set_chat_history(updated_history)
    return updated_history


def build_chat_messages(message: str, history: List[Message]):
    last_event = get_last_event()
    messages = [
        {
            "role": "system",
            "content": (
                "Voce e o assistente do sistema AgroVision. "
                "Responda em portugues, com base no historico da conversa e no ultimo evento detectado."
            ),
        }
    ]

    if last_event:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Ultimo evento detectado:\n"
                    f"- ID: {last_event['id']}\n"
                    f"- Horario: {last_event['event_time']}\n"
                    f"- Objeto: {last_event['label']}\n"
                    f"- Confianca: {last_event['confidence']:.2f}\n"
                    f"- Imagem: {last_event['image_path'] or 'nao disponivel'}"
                ),
            }
        )
    else:
        messages.append(
            {
                "role": "system",
                "content": "Ainda nao ha eventos detectados no banco de dados.",
            }
        )

    messages.extend(message_to_dict(item) for item in history)
    messages.append({"role": "user", "content": message})
    return messages


def ask_ollama(message: str, history: List[Message]):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": build_chat_messages(message, history),
        "stream": False,
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=(10, OLLAMA_TIMEOUT))
    response.raise_for_status()
    data = response.json()
    answer = data.get("message", {}).get("content", "").strip()
    return answer


def set_visibility_state(status: str, details: str):
    with visibility_lock:
        visibility_state["status"] = status
        visibility_state["details"] = details
        visibility_state["updated_at"] = time.strftime("%H:%M:%S")


def visibility_worker():
    while True:
        time.sleep(5)
        with latest_frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            continue

        try:
            results = model(frame, device=device, verbose=False)
            person_count = 0

            for r in results:
                for box in r.boxes:
                    label = model.names[int(box.cls)]
                    if label == "person":
                        person_count += 1

            if person_count > 0:
                detail = f"{person_count} pessoa(s) detectada(s) na imagem."
                set_visibility_state("pessoa detectada", detail)
            else:
                set_visibility_state("sem pessoa", "Nenhuma pessoa detectada na imagem.")
        except Exception as exc:
            set_visibility_state("indisponivel", f"Falha ao analisar a camera: {exc}")


def stream_ollama(message: str, history: List[Message]):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": build_chat_messages(message, history),
        "stream": True,
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=(10, OLLAMA_TIMEOUT), stream=True)
    response.raise_for_status()

    answer_parts: List[str] = []

    def event_stream():
        try:
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue

                data = json.loads(raw_line.decode("utf-8"))
                chunk = data.get("message", {}).get("content", "")

                if chunk:
                    answer_parts.append(chunk)
                    yield json.dumps({"type": "chunk", "content": chunk}) + "\n"

            final_answer = "".join(answer_parts).strip()
            updated_history = append_chat_exchange(message, final_answer, history)
            yield json.dumps(
                {
                    "type": "done",
                    "history": [message_to_dict(item) for item in updated_history],
                }
            ) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "error": str(exc)}) + "\n"
        finally:
            response.close()

    return event_stream()


def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        with latest_frame_lock:
            global latest_frame
            latest_frame = frame.copy()

        results = model(frame, device=device, verbose=False)
        annotated_frame = results[0].plot()

        for r in results:
            if len(r.boxes) > 0:
                label = model.names[int(r.boxes[0].cls)]
                conf = float(r.boxes[0].conf)
                conn = sqlite3.connect(DB_PATH)
                conn.execute("INSERT INTO events (label, confidence) VALUES (?, ?)", (label, conf))
                conn.commit()
                conn.close()

        ok, buffer = cv2.imencode(".jpg", annotated_frame)
        if not ok:
            continue
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    cap.release()


visibility_thread = threading.Thread(target=visibility_worker, daemon=True)
visibility_thread.start()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    context = {
        "request": request,
        "initial_history": [message_to_dict(item) for item in get_chat_history()],
    }
    return templates.TemplateResponse(request=request, name="index.html", context=context)


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/visibility")
async def get_visibility():
    with visibility_lock:
        return dict(visibility_state)


@app.get("/agro-news")
async def get_agro_news():
    return scrape_agro_news()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    base_history = req.history or get_chat_history()
    answer = ask_ollama(req.message, base_history)
    updated_history = append_chat_exchange(req.message, answer, base_history)
    return ChatResponse(answer=answer, history=updated_history)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    base_history = req.history or get_chat_history()
    return StreamingResponse(stream_ollama(req.message, base_history), media_type="application/x-ndjson")
