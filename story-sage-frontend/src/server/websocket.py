import asyncio
import websockets
import json
import base64
from audio_reader import AudiobookReaderContinuous
import os
from typing import Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store active connections
connections: Dict[str, websockets.WebSocketServerProtocol] = {}

async def process_audio(websocket, path):
    """Handle WebSocket connection for audio processing."""
    try:
        # Get initial configuration
        message = await websocket.recv()
        config = json.loads(message)
        
        pdf_path = config['pdf_path']
        output_dir = config['output_dir']
        language = config.get('language', 'en')
        
        # Initialize audio reader
        reader = AudiobookReaderContinuous()
        
        # Create a callback to send audio data through WebSocket
        async def send_audio_chunk(chunk):
            try:
                # Convert audio chunk to base64
                audio_base64 = base64.b64encode(chunk).decode('utf-8')
                await websocket.send(json.dumps({
                    'type': 'audio',
                    'audio': audio_base64
                }))
            except Exception as e:
                print(f"Error sending audio chunk: {e}")
        
        # Process the book with streaming
        await reader.process_book_streaming(
            pdf_path=pdf_path,
            output_dir=output_dir,
            language=language,
            audio_callback=send_audio_chunk
        )
        
        # Send completion message
        await websocket.send(json.dumps({
            'type': 'complete',
            'wavUrl': f"{output_dir}/output.wav"  # Add the WAV file URL
        }))
        
    except Exception as e:
        logger.error(f"Error in process_audio: {e}")
        try:
            await websocket.send(json.dumps({
                'type': 'error',
                'message': str(e)
            }))
        except:
            pass
    finally:
        # Cleanup
        if 'reader' in locals():
            reader.cleanup()

async def main():
    """Start WebSocket server."""
    server = await websockets.serve(
        process_audio,
        "localhost",
        8000
    )
    logger.info("WebSocket server started on ws://localhost:8000")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())