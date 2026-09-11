from dotenv import load_dotenv
import os
import certifi

# Clean environment to prevent stale cached keys
for key in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "TAVILY_API_KEY", "LANGSMITH_API_KEY"]:
    if key in os.environ:
        del os.environ[key]

load_dotenv(override=True)

# Synchronize API keys so langchain uses the active one
active_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if active_key:
    os.environ["GOOGLE_API_KEY"] = active_key
    os.environ["GEMINI_API_KEY"] = active_key

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import json
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    AIMessageChunk,
    ToolMessage
)

from agent import get_agent
from database import (
    init_db,
    save_chat_message,
    get_chat_history,
    create_or_update_conversation,
    list_conversations
)

from rag import add_document_to_rag
from tools import set_current_thread_id


app = FastAPI(title="AVATAR - Real-World Problem to Verified Action Assistant")

templates = Jinja2Templates(directory="templates")

Path("uploads").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

init_db()


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/conversations")
async def conversations():
    items = list_conversations()
    return {
        "conversations": [
            {
                "thread_id": item.thread_id,
                "title": item.title,
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat()
            }
            for item in items
        ]
    }


@app.get("/history/{thread_id}")
async def history(thread_id: str):
    messages = get_chat_history(thread_id)
    return {
        "messages": [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]
    }


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    thread_id: str = Form(...)
):
    try:
        allowed_extensions = [".pdf", ".docx", ".txt", ".md", ".py", ".csv"]

        filename = file.filename or "uploaded_file"
        suffix = Path(filename).suffix.lower()

        if suffix not in allowed_extensions:
            return JSONResponse(
                {
                    "success": False,
                    "message": "Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV."
                },
                status_code=400
            )

        file_id = str(uuid.uuid4())[:8]
        safe_filename = filename.replace(" ", "_")
        file_path = f"uploads/{file_id}_{safe_filename}"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        create_or_update_conversation(thread_id, f"Uploaded {filename}")

        result = add_document_to_rag(
            file_path=file_path,
            thread_id=thread_id
        )

        return JSONResponse({
            "success": True,
            "filename": result["filename"],
            "chunks": result["chunks"],
            "message": f"Uploaded '{result['filename']}' and indexed {result['chunks']} chunks into verified knowledge base."
        })

    except Exception as e:
        return JSONResponse(
            {
                "success": False,
                "message": str(e)
            },
            status_code=500
        )


@app.post("/upload_image")
async def upload_image(
    file: UploadFile = File(...),
    thread_id: str = Form(...)
):
    """
    Upload notice, form, screenshot, or photo for multimodal visual analysis.
    """
    try:
        allowed_extensions = [".png", ".jpg", ".jpeg", ".webp"]
        filename = file.filename or "uploaded_image.png"
        suffix = Path(filename).suffix.lower()

        if suffix not in allowed_extensions:
            return JSONResponse(
                {
                    "success": False,
                    "message": "Please upload a valid image (PNG, JPG, JPEG, or WEBP)."
                },
                status_code=400
            )

        file_id = str(uuid.uuid4())[:8]
        safe_filename = filename.replace(" ", "_")
        stored_name = f"img_{file_id}_{safe_filename}"
        file_path = f"uploads/{stored_name}"

        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        return JSONResponse({
            "success": True,
            "filename": filename,
            "file_path": file_path,
            "url": f"/uploads/{stored_name}",
            "message": f"Image '{filename}' ready for multimodal verification."
        })
    except Exception as e:
        return JSONResponse(
            {"success": False, "message": str(e)},
            status_code=500
        )


def sse_data(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def should_stream_chunk(chunk, metadata) -> bool:
    """
    Only stream normal AI text chunks from the action_planner / chatbot node.
    Filter out internal node messages, tools, and JSON classification outputs.
    """
    metadata = metadata or {}
    node_name = str(metadata.get("langgraph_node", "")).lower()

    if node_name and node_name not in ["action_planner", "chatbot"]:
        return False

    if isinstance(chunk, ToolMessage):
        return False

    if not isinstance(chunk, (AIMessage, AIMessageChunk)):
        return False

    if getattr(chunk, "tool_calls", None):
        return False

    if getattr(chunk, "invalid_tool_calls", None):
        return False

    additional_kwargs = getattr(chunk, "additional_kwargs", {}) or {}
    if additional_kwargs.get("tool_calls"):
        return False

    return True


def extract_text_from_chunk(chunk) -> str:
    content = getattr(chunk, "content", "")

    if not content:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    text_parts.append(item["content"])
        return "".join(text_parts)

    return ""


@app.post("/chat/stream")
async def chat_stream(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(
            {"error": "Invalid JSON body."},
            status_code=400
        )

    user_message = data.get("message", "")
    thread_id = data.get("thread_id", "default")
    selected_model = data.get("model", "gemini-3.6-flash")
    language = data.get("language", "English")
    image_path = data.get("image_path", None)

    if not user_message.strip() and not image_path:
        return JSONResponse(
            {"error": "Message or image is required."},
            status_code=400
        )

    try:
        agent = get_agent(selected_model)
    except Exception as e:
        return JSONResponse(
            {"detail": str(e)},
            status_code=400
        )

    display_message = user_message
    if image_path:
        img_name = Path(image_path).name
        if display_message:
            display_message = f"![Attached Notice](/uploads/{img_name})\n\n{display_message}"
        else:
            display_message = f"![Attached Notice](/uploads/{img_name})\n\nPlease analyze this notice/document and provide a verified action plan."
            user_message = "Analyze this attached notice/document and provide a structured, verified action plan."

    create_or_update_conversation(thread_id, user_message)
    save_chat_message(thread_id, "user", display_message)

    set_current_thread_id(thread_id)

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    async def event_generator():
        final_answer = ""

        try:
            db_messages = get_chat_history(thread_id)
            formatted_messages = []
            for msg in db_messages:
                if msg.role == "user":
                    formatted_messages.append(HumanMessage(content=msg.content))
                else:
                    formatted_messages.append(AIMessage(content=msg.content))

            inputs = {
                "messages": formatted_messages,
                "user_input": user_message,
                "language": language,
                "image_path": image_path,
                "intent": "",
                "urgency": "NORMAL",
                "entities": {},
                "route": "DIRECT",
                "rag_evidence": "",
                "web_evidence": "",
                "image_evidence": "",
                "verification_notes": "",
                "action_plan": "",
                "sources": []
            }

            async for chunk, metadata in agent.astream(
                inputs,
                config=config,
                stream_mode="messages"
            ):
                if not should_stream_chunk(chunk, metadata):
                    continue

                token = extract_text_from_chunk(chunk)
                if token:
                    final_answer += token
                    yield sse_data({"token": token})

            if final_answer.strip():
                save_chat_message(thread_id, "assistant", final_answer)

            yield sse_data({"done": True})

        except Exception as e:
            yield sse_data({"error": str(e)})
            yield sse_data({"done": True})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True,
        reload_excludes=[
            "data/*",
            "chroma_db/*",
            "uploads/*",
            "*.db",
            "*.sqlite",
            "*.sqlite-journal",
            "__pycache__/*",
        ]
    )