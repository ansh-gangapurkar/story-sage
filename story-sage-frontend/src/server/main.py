import logging
import os
import json
import uuid
from fastapi import FastAPI, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
from typing import Dict
import base64
from audio_reader import AudiobookReaderContinuous
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories at startup
temp_dir = os.path.join(os.getcwd(), "temp")
os.makedirs(temp_dir, exist_ok=True)
output_dir = os.path.join(temp_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# Mount the temp directory for static file serving
app.mount("/audio", StaticFiles(directory=temp_dir), name="audio")

# Store active WebSocket connections
connections: Dict[str, WebSocket] = {}

@app.post("/upload")
async def upload_file(file: UploadFile):
    try:
        # Create temp directory if it doesn't exist
        temp_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(temp_dir, exist_ok=True)

        # Save the uploaded file
        file_path = os.path.join(temp_dir, f"{file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Create output directory
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        return {
            "success": True,
            "file_path": file_path,
            "output_dir": output_dir
        }
    except Exception as e:
        return {"error": str(e)}, 500

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    connections[client_id] = websocket
    
    try:
        data = await websocket.receive_json()
        pdf_path = data["pdf_path"]
        output_dir = data["output_dir"]
        language = data.get("language", "en")

        reader = AudiobookReaderContinuous()

        # Process without streaming
        await reader.process_book_streaming(
            pdf_path=pdf_path,
            output_dir=output_dir,
            language=language,
            audio_callback=None  # Don't stream audio chunks
        )

        # Send completion with correct WAV file URL
        relative_path = os.path.relpath(
            os.path.join(output_dir, "book_continuous.wav"), 
            temp_dir
        )
        await websocket.send_json({
            "type": "complete",
            "wavUrl": f"/audio/{relative_path}"
        })

    except Exception as e:
        logger.error(f"Error in websocket connection: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        if client_id in connections:
            del connections[client_id]
        if 'reader' in locals():
            reader.cleanup()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)