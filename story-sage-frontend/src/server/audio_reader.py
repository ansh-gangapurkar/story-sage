import os
import PyPDF2
import google.generativeai as genai
from cartesia import Cartesia
from typing import Dict, List
import json
import asyncio
from dotenv import load_dotenv
import logging
import atexit
import requests
import websockets
import uuid
import pyaudio
import soundfile as sf
import base64

voicesUrl = "https://api.cartesia.ai/voices/"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudiobookReaderContinuous:
    def __init__(self):
        # Initialize API keys and clients
        self.cartesia_api_key = os.environ.get("CARTESIA_API_KEY")
        self.gemini_api_key = os.environ.get("GOOGLE_API_KEY")

        if not self.cartesia_api_key:
            raise ValueError("CARTESIA_API_KEY environment variable is not set")
        if not self.gemini_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")

        self.cartesia_client = Cartesia(api_key=self.cartesia_api_key)
        genai.configure(api_key=self.gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')

        # Get available voices from Cartesia API
        headers = {
            "Cartesia-Version": "2024-06-10",
            "X-API-Key": self.cartesia_api_key
        }
        response = requests.get(voicesUrl, headers=headers)
        self.available_voices = response.json()

        # Create voices prompt for Gemini
        self.voices_prompt = "Available voices:\n"
        for voice in self.available_voices:
            self.voices_prompt += f"ID: {voice['id']}, Description: {voice['description']}, Language: {voice['language']}\n"

        # Emotion mapping from Gemini to Cartesia
        self.emotion_mapping = {
            "happy": "positivity",
            "sad": "sadness",
            "angry": "anger",
            "excited": "positivity",
            "worried": "sadness",
            "calm": None,
            "none": None
        }

        # Initialize PyAudio for audio streaming and websocket for continuous audio
        self.audio_data = bytearray()
        self.p = pyaudio.PyAudio()
        self.rate = 22050
        self.stream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=self.rate, output=True)

        atexit.register(self.cleanup)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            logger.error(f"Error reading PDF file: {str(e)}")
            raise

    def analyze_text_and_assign_voices_with_gemini(self, text: str, language: str) -> List[Dict]:
        """
        Use Gemini API to analyze text, identify speakers, and assign voice IDs in one step.
        Returns a list of dictionaries containing speaker, text, and voice ID information.
        """
        try:
            prompt = f"""
You are provided with a text excerpt from a story, which may be formatted as a play script. Your task is to process the text as follows, adhering strictly to the instructions below:

1. **Sentence Segmentation & Speaker Identification & Translation (if needed):**
   - Break the text into individual sentences based on standard punctuation (e.g., periods, exclamation marks, question marks).
   - For each sentence, identify the speaker. If the speaker is not explicitly named (e.g., in dialogue tags like "said John"), infer it from context (e.g., alternating dialogue patterns or narrative cues). Use "narrator" for non-dialogue text unless context suggests otherwise.
   - **Ignore Physical Directions and Stage Notes**: Exclude any text enclosed in parentheses (e.g., "(waving goodbye)", "(takes a deep breath)") or lines that describe actions, gestures, or scene settings without spoken content. Only process text intended as spoken dialogue or narrative.
   - Determine the original language of the text. If it is NOT '{language}', translate each sentence accurately into '{language}' while preserving meaning, tone, and intent. If the original language is already '{language}', the translated text should be identical to the original text.

2. **Voice ID Assignment:**
   - For each identified speaker:
     - **Infer Personality and Language**: Analyze the speaker’s dialogue (or narrative style for the narrator) to infer their personality traits (e.g., cheerful, stern, mysterious, calm) and confirm their language aligns with '{language}' post-translation. Ignore any personality cues from excluded stage directions.
     - **Language Constraint**: From the list of available voices, select ONLY those where the 'language' field exactly matches '{language}'. Exclude any voice that does not support '{language}', even if it seems otherwise suitable.
     - **Personality Matching**: Choose the voice ID whose 'description' most closely matches the inferred personality of the speaker. Prioritize voices with descriptors that align with the character’s tone, mood, or traits (e.g., "warm and soothing" for a kind character, "deep and authoritative" for a leader). If multiple voices match the language and personality, select the one with the closest fit based on the description’s specificity.
     - **Narrator Handling**: For "narrator" speakers, default to a voice with a neutral, clear, and versatile description (e.g., "calm and articulate") unless the narrative tone suggests a specific personality (e.g., "mysterious" for a suspenseful story).
     - **Fallback**: If no voice perfectly matches both the language and personality, choose the voice that matches '{language}' and has the most neutral description (e.g., "clear and neutral") to ensure usability.
   - Ensure each speaker’s voice ID is consistent across all their sentences unless a deliberate change is justified by the story (e.g., a character’s mood shift explicitly described in the dialogue).

3. **Emotion Analysis:**
   - For each sentence, analyze the emotional content and assign one or more of these emotions:
     - "happy"
     - "sad"
     - "angry"
     - "none"
     - "excited"
     - "worried"
     - "calm"
   - Base the emotion on the sentence content, punctuation, and context.

4. **Output Requirements:**
   - Return a valid JSON array where each element is a JSON object with these fields:
     - "speaker": the identified speaker's name
     - "original_text": the untranslated text
     - "translated_text": the text in '{language}'
     - "voice_id": the assigned Cartesia voice ID
     - "emotions": array of emotion strings for this segment
   - Do NOT include additional text, comments, or formatting outside the JSON array. The output must be pure JSON.
   - Validate that every "voice_id" corresponds to a voice from the provided list and supports '{language}'.

**Available Voices:**
{self.voices_prompt}

**Analyze the Text Below and Produce the JSON Output:**
{text}
"""

            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()

            def clean_json_response(text):  # Helper function for JSON cleaning
                cleaned = text.replace("```json", "").replace("```python", "").replace("```", "").strip()
                return cleaned.strip()

            try:
                return_text = json.loads(response_text)
            except json.JSONDecodeError:
                cleaned_text = clean_json_response(response_text)
                try:
                    return_text = json.loads(cleaned_text)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse Gemini response: {response_text}")
                    logger.error(f"Cleaned version: {cleaned_text}")
                    raise

            # Process the returned JSON to use "translated_text"
            processed_segments = []
            for segment in return_text:
                processed_segments.append({
                    "speaker": segment.get("speaker"),
                    "original_text": segment.get("original_text"),  # Keep original text in metadata
                    "text": segment.get("translated_text"),        # Use translated text for TTS
                    "voice_id": segment.get("voice_id"),
                    "emotions": segment.get("emotions", ["none"])  # Default to "none" if no emotions
                })
            return processed_segments

        except Exception as e:
            logger.error(f"Error in Gemini API call: {str(e)}")
            raise

    async def generate_audio(self, text: str, voice_id: str, output_file: str, context_id: str, language: str, continue_stream: bool, emotions: List[str] = None):
        """Generate audio for a piece of text using Cartesia API with WebSocket streaming and context ID."""
        try:
            # Map emotions to Cartesia-supported emotions
            mapped_emotions = []
            if emotions:
                for emotion in emotions:
                    mapped_emotion = self.emotion_mapping.get(emotion)
                    if mapped_emotion:
                        mapped_emotions.append(mapped_emotion)

            # Create WebSocket request with continuation parameters
            request = {
                "model_id": "sonic",
                "voice": {
                    "mode": "id",
                    "id": voice_id
                },
                "language": language,
                "context_id": context_id,
                "stream": True,
                "transcript": text,
                "continue": continue_stream,  # Add continuation flag
                "output_format": {
                    "container": "raw",
                    "encoding": "pcm_f32le",
                    "sample_rate": 22050
                },
                "_experimental_voice_controls": {
                    "speed": 0,
                    "emotion": mapped_emotions
                }
            }

            # Use WebSocket for streaming
            async with websockets.connect("wss://api.cartesia.ai/tts/websocket?cartesia_version=2024-06-10&api_key=sk_car_NaJWSjmUaaXoBfSrfOXnX") as websocket:
                await websocket.send(json.dumps(request))
                async for message in websocket:
                    output = json.loads(message)
                    try:
                        buffer = base64.b64decode(output["data"])  # Decode the base32-encoded audio data
                    except Exception:
                        break

                    # Write the audio data to the stream
                    self.stream.write(buffer)
                    self.audio_data.extend(buffer)

            logger.info(f"Generated audio with emotions {mapped_emotions} (Context ID: {context_id}): {output_file}")

        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}")
            raise

    async def process_book_streaming(self, pdf_path: str, output_dir: str, language: str, audio_callback=None) -> None:
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            text = self.extract_text_from_pdf(pdf_path)
            segments_with_voices = self.analyze_text_and_assign_voices_with_gemini(text, language=language)
            
            # Process segments with context continuation
            context_id = str(uuid.uuid4())  # Single context ID for the entire book
            
            for i, segment in enumerate(segments_with_voices):
                # Set continue_stream flag for all but the last segment
                continue_stream = i < (len(segments_with_voices) - 1)
                
                await self.generate_audio(
                    text=segment["text"],
                    voice_id=segment["voice_id"],
                    output_file=None,  # No need to output individual raw files
                    context_id=context_id,
                    language=language,
                    continue_stream=continue_stream,
                    emotions=segment["emotions"]
                )
            
            # Write the collected audio data to file
            audio_data_file = os.path.join(output_dir, "collected_audio.raw")
            with open(audio_data_file, "wb") as f:
                f.write(self.audio_data)

            # Clean up audio components
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            
            # Convert the collected audio data to a WAV file
            final_wav_file = os.path.join(output_dir, "book_continuous.wav")
            self.convert_raw_to_wav(audio_data_file, final_wav_file)
            
            # Clean up the raw audio data file
            os.remove(audio_data_file)

        except Exception as e:
            logger.error(f"Error processing book: {str(e)}")
            raise

    @staticmethod
    def convert_raw_to_wav(raw_filepath, wav_filepath, sample_rate=22050, channels=1, subtype='FLOAT'):
        """Static method to convert raw PCM to WAV using soundfile."""
        try:
            raw_data = None
            with open(raw_filepath, 'rb') as raw_file:
                raw_data = raw_file.read()

            if not raw_data:
                logger.error(f"Raw data is empty from {raw_filepath}")
                return

            # Assuming 32-bit float raw data, reshape it to a 1D array of floats
            import numpy as np
            float_data = np.frombuffer(raw_data, dtype=np.float32)

            sf.write(wav_filepath, float_data, sample_rate, subtype='FLOAT')  # Explicitly use FLOAT subtype
            logger.info(f"Converted {raw_filepath} to {wav_filepath} using soundfile")

        except Exception as e:
            logger.error(f"Error converting {raw_filepath} to WAV using soundfile: {e}")
            raise

    def cleanup(self):
        """Cleanup resources before exit."""
        try:
            del self.gemini_model
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")

if __name__ == "__main__":
    load_dotenv()
    try:
        reader = AudiobookReaderContinuous()
        pdf_path = "/Users/ansh_is_g/Documents/story-sage/story-sage/The-Tortoise-and-the-Hare.pdf"
        output_dir = "output_audio_continuous"
        logger.info("Starting continuous book processing with Context ID...")
        
        # Create a simple callback for testing
        async def test_callback(audio_chunk):
            pass  # In test mode, we don't need to do anything with the audio chunks
        
        # Use process_book_streaming
        asyncio.run(reader.process_book_streaming(
            pdf_path=pdf_path,
            output_dir=output_dir,
            language="en",  # Default to English
            audio_callback=test_callback
        ))
        
        logger.info("Continuous book processing completed successfully!")
    except Exception as e:
        logger.error(f"Error during continuous execution: {e}")
        raise
    finally:
        if 'reader' in locals():
            reader.cleanup()
