from cartesia import Cartesia
import pyaudio
import os

client = Cartesia(
    api_key=os.getenv("CARTESIA_API_KEY"),
)
voice_id = "a0e99841-438c-4a64-b679-ae501e7d6091"
transcript = "Hello! Welcome to Cartesia"

# You can check out our models at https://docs.cartesia.ai/getting-started/available-models
model_id = "sonic"

p = pyaudio.PyAudio()
rate = 22050

stream = None

# Set up the websocket connection
ws = client.tts.websocket()

# Generate and stream audio using the websocket
for output in ws.send(
    model_id=model_id,
    transcript=transcript,
    voice_id=voice_id,
    stream=True,
    output_format={
        "container": "raw",
        "encoding": "pcm_f32le", 
        "sample_rate": 22050
    },
):
    buffer = output["audio"]

    if not stream:
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=rate, output=True)

    # Write the audio data to the stream
    stream.write(buffer)

stream.stop_stream()
stream.close()
p.terminate()

ws.close()  # Close the websocket connection