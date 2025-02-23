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
import wave  # Import wave for WAV conversion
import soundfile as sf

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
        }

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

1. **Sentence Segmentation, Speaker Identification, & Translation (if needed):**
   - Break the text into individual sentences based on standard punctuation (e.g., periods, exclamation marks, question marks).
   - Identify the speaker for each sentence. If the speaker is not explicitly named (e.g., "said John"), infer it from context (e.g., alternating dialogue patterns or narrative cues). Use "narrator" for non-dialogue text unless context suggests otherwise.
   - **Exclude Physical Directions and Stage Notes**: Omit any text enclosed in parentheses (e.g., "(waving goodbye)", "(takes a deep breath)") or lines describing actions, gestures, or scene settings without spoken content. Only process spoken dialogue or narrative text.
   - Detect the original language of the text. If it is NOT '{language}', translate each sentence with 100% accuracy into '{language}' with native text, ensuring no meaning, tone, or intent is lost or altered. If the original text is already in '{language}', retain it as-is.

2. **Voice ID Assignment:**
   - For each identified speaker:
     - **Infer Personality and Confirm Language**: Analyze the speaker’s dialogue and gender (or the narrator’s style) to infer their personality traits (e.g., cheerful, stern, mysterious, calm) and confirm the final text is in '{language}'.
     - **Language Compliance**: Select ONLY voice IDs where the 'language' field matches '{language}' exactly. Exclude voices that do not explicitly support '{language}'.
     - **Personality Matching**: Choose the voice ID whose 'description' aligns most closely with the speaker’s inferred personality. Prioritize matches based on tone, mood, and traits (e.g., "warm and soothing" for a kind character, "deep and authoritative" for a leader). If multiple voices match, select the one with the most specific alignment.
     - **Narrator Voice**: For "narrator" speakers, default to a neutral, clear, and versatile voice (e.g., "calm and articulate") unless the narrative tone suggests otherwise (e.g., "mysterious" for suspense).
     - **Fallback Rule**: If no voice perfectly matches both language and personality, choose a voice that matches '{language}' with a neutral description (e.g., "clear and neutral") to maintain consistency.
   - Ensure voice ID consistency for each speaker throughout the text unless a significant shift in tone or mood, explicitly indicated by the dialogue, justifies a change.

3. **Emotion Analysis:**
   - Analyze the emotional content of each sentence and assign one or more of the following emotions:
     - "happy"
     - "sad"
     - "angry"
     - "excited"
     - "worried"
   - Base this analysis on sentence content, punctuation, and contextual cues. 

4. **Output Requirements:**
   - Return a valid JSON array where each element contains:
     - "speaker": the identified speaker’s name
     - "original_text": the untranslated text
     - "translated_text": the text in '{language}'
     - "voice_id": the assigned Cartesia voice ID
     - "emotions": array of identified emotions for the sentence
   - **No Extra Content**: Only return the JSON array—do NOT include comments, explanations, or additional formatting.
   - Ensure that each "voice_id" corresponds to an available voice that supports '{language}'.

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

    async def generate_audio(self, text: str, voice_id: str, output_file: str, context_id: str, language: str, emotions: List[str] = None):
        """Generate audio for a piece of text using Cartesia API with WebSocket streaming and context ID."""
        try:
            ws_url = "wss://api.cartesia.ai/tts/websocket?cartesia_version=2024-06-10"
            headers = {
                "Cartesia-Version": "2024-06-10",
                "X-API-Key": self.cartesia_api_key
            }

            p = pyaudio.PyAudio()
            rate = 22050
            stream = None
            
            ws = self.cartesia_client.tts.websocket()
            audio_data = bytearray()

            # Map emotions to Cartesia-supported emotions
            if emotions:
                mapped_emotions = []
                for emotion in emotions:
                    mapped_emotion = self.emotion_mapping.get(emotion)
                    if mapped_emotion:  # Only append if there's a valid mapping
                        mapped_emotions.append(mapped_emotion)
            else:
                mapped_emotions = []

            for output in ws.send(
                model_id="sonic",
                transcript=text,
                voice_id=voice_id,
                stream=True,  # Enable streaming
                output_format={
                    "container": "raw",
                    "encoding": "pcm_f32le",
                    "sample_rate": 22050
                },
                language=language,
                context_id=context_id,  # Keep context_id for continuity
                _experimental_voice_controls={
                    "speed": 0,
                    "emotion": mapped_emotions  # Use mapped emotions
                }
            ):
                buffer = output["audio"]

                if not stream:
                    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=rate, output=True)

                stream.write(buffer)
                audio_data.extend(buffer)

            stream.stop_stream()
            stream.close()
            p.terminate()

            # Write the collected audio data to file
            with open(output_file, "wb") as f:
                f.write(audio_data)

            logger.info(f"Generated audio file with emotions {mapped_emotions} (Context ID: {context_id}): {output_file}")

        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}")
            raise

    async def process_book_streaming(self, pdf_path: str, output_dir: str, language: str, audio_callback=None) -> None:
        try:
            # Create the output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Create a subdirectory for raw files
            raw_dir = os.path.join(output_dir, "raw")
            os.makedirs(raw_dir, exist_ok=True)
            
            text = self.extract_text_from_pdf(pdf_path)
            segments_with_voices = self.analyze_text_and_assign_voices_with_gemini(text, language=language)
            
            # Process segments and save RAW files
            segment_raw_files = []
            context_id = str(uuid.uuid4())
            
            for i, segment in enumerate(segments_with_voices):
                # Save raw files in the raw subdirectory
                raw_filepath = os.path.join(raw_dir, f"segment_{i}.raw")
                
                await self.generate_audio(
                    text=segment["text"],
                    voice_id=segment["voice_id"],
                    output_file=raw_filepath,
                    context_id=context_id,
                    language=language,
                    emotions=segment["emotions"]  # Pass emotions directly from segment
                )
                segment_raw_files.append(raw_filepath)
                
            # Concatenate all RAW files
            final_raw_file = os.path.join(raw_dir, "book_continuous.raw")
            with open(final_raw_file, 'wb') as outfile:
                for raw_file in segment_raw_files:
                    with open(raw_file, 'rb') as infile:
                        outfile.write(infile.read())
                    
            # Convert to WAV in the main output directory
            final_wav_file = os.path.join(output_dir, "book_continuous.wav")
            self.convert_raw_to_wav(final_raw_file, final_wav_file)
            
            # Clean up raw files
            for raw_file in segment_raw_files:
                os.remove(raw_file)
            os.remove(final_raw_file)
            os.rmdir(raw_dir)

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