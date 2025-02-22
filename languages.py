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
import wave # Import wave for WAV conversion

voicesUrl = "https://api.cartesia.ai/voices/"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudiobookReaderContinuous:
    def __init__(self):
        # Initialize API keys and clients - same as before
        self.cartesia_api_key = os.environ.get("CARTESIA_API_KEY")
        self.gemini_api_key = os.environ.get("GOOGLE_API_KEY")

        if not self.cartesia_api_key:
            raise ValueError("CARTESIA_API_KEY environment variable is not set")
        if not self.gemini_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")

        self.cartesia_client = Cartesia(api_key=self.cartesia_api_key)
        genai.configure(api_key=self.gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash')

        # Get available voices from Cartesia API - same as before
        headers = {
            "Cartesia-Version": "2024-06-10",
            "X-API-Key": self.cartesia_api_key
        }
        response = requests.get(voicesUrl, headers=headers)
        self.available_voices = response.json()

        # Create voices prompt for Gemini - same as before
        self.voices_prompt = "Available voices:\n"
        for voice in self.available_voices:
            self.voices_prompt += f"ID: {voice['id']}, Description: {voice['description']}, Language: {voice['language']}\n"

        atexit.register(self.cleanup)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file - same as before."""
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
You are provided with a text excerpt from a story. Your task is to process the text as follows:

1. **Sentence Segmentation & Speaker Identification & Translation (if needed):**
   - Break the text into individual sentences.
   - Identify the speaker for each sentence.
   - **If the language of the text is NOT '{language}', accurately translate each sentence into '{language}'.  If no translation is needed, the translated text should be the same as the original text.**

2. **Voice ID Assignment:**
   - For each identified speaker, infer their personality and language from their dialogue (or narrative context for narrator).
   - Based on the inferred personality and speaker language, choose the most fitting voice ID from the provided list of available voices. **Prioritize voices that are suitable for the target language '{language}' if translation was performed.**
   - If a speaker is "narrator", choose a default narrator voice, unless the context suggests a specific tone.

3. **Output Requirements:**
   - Return a valid JSON array where each element is a JSON object with the following fields:
     - "speaker": the name of the identified speaker.
     - "original_text": the original sentence text from the PDF.
     - "translated_text": **the sentence text, translated into '{language}' if necessary, otherwise the same as "original_text".**
     - "voice_id": the Cartesia voice ID assigned to this speaker.
   - Ensure the output is ONLY the JSON array, without extra text or formatting.

Available voices:
{self.voices_prompt}

Analyze the text below and produce the JSON output:
{text}
"""

            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()

            def clean_json_response(text): # Helper function for JSON cleaning - moved here for better encapsulation
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
                    "original_text": segment.get("original_text"), # Keep original text in metadata
                    "text": segment.get("translated_text"),      # Use translated text for TTS
                    "voice_id": segment.get("voice_id")
                })
            return processed_segments


        except Exception as e:
            logger.error(f"Error in Gemini API call: {str(e)}")
            raise


    async def generate_audio(self, text: str, voice_id: str, output_file: str, context_id: str, continue_stream: bool, language: str):
        """Generate audio for a piece of text using Cartesia API with WebSocket streaming and context ID."""
        try:
            ws_url = "wss://api.cartesia.ai/tts/websocket?cartesia_version=2024-06-10&api_key=sk_car_NaJWSjmUaaXoBfSrfOXnX" #API KEY IS HARDCODED HERE - SHOULD BE REMOVED

            headers = {
                "Cartesia-Version": "2024-06-10",
                "X-API-Key": self.cartesia_api_key
            }

            p = pyaudio.PyAudio()
            rate = 22050
            stream = None

            async with websockets.connect(ws_url) as websocket:
                # Prepare the initial request with context_id and continue flag
                request = {
                    "model_id": "sonic",
                    "voice": {
                        "mode": "id",
                        "id": voice_id
                    },
                    "language": language, # Use the language parameter here!
                    "context_id": context_id,
                    "transcript": text[:50], # Send initial 50 chars as per documentation
                    "continue": continue_stream, # Set continue flag based on input
                    "output_format": {
                        "container": "raw",
                        "encoding": "pcm_f32le",
                        "sample_rate": 22050,
                    }
                }
                ws = self.cartesia_client.tts.websocket()

                audio_data = bytearray()

                # Generate and stream audio using the websocket
                for output in ws.send(
                    model_id="sonic",
                    transcript=text,
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

                    audio_data.extend(buffer)

                stream.stop_stream()
                stream.close()
                p.terminate()

                # Write the collected audio data to file
                with open(output_file, "wb") as f:
                    f.write(audio_data)

                logger.info(f"Generated audio file using WebSocket (Context ID: {context_id}): {output_file}")

        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}")
            raise


    async def process_book(self, pdf_path: str, output_dir: str):
        """Main function to process the book and generate continuous audio file."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info("Extracting text from PDF...")
            text = self.extract_text_from_pdf(pdf_path)

            language = input("Enter the language to read: ") # Get language input here

            logger.info("Analyzing text with Gemini API and assigning voice IDs...")
            segments_with_voices = self.analyze_text_and_assign_voices_with_gemini(text, language=language) # Pass language to Gemini

            metadata = { # Directly create metadata in the desired format
                "segments": segments_with_voices,
                "unique_speakers": {} # We can still keep unique speakers if needed for metadata
            }
            speakers_dict = {}
            for segment in segments_with_voices:
                speakers_dict[segment['speaker']] = segment['voice_id'] # Directly map speaker to voice_id
            metadata['unique_speakers'] = speakers_dict


            logger.info("Generating audio segments using WebSocket with Context ID...")
            overall_audio_data = bytearray() # Accumulate audio data for all segments
            context_id = str(uuid.uuid4()) # Generate a single context_id for the whole book

            segment_raw_files = [] # Keep track of raw segment files to concatenate later

            for i, segment in enumerate(segments_with_voices): # Use segments with voice_ids directly
                try:
                    text = segment["text"]
                    speaker = segment["speaker"]
                    voice_id = segment["voice_id"] # Voice ID is already assigned by Gemini
                    continue_stream = i < len(segments_with_voices) - 1 # Continue stream for all but the last segment

                    if not voice_id:
                        logger.error(f"No voice ID found for speaker: {speaker}")
                        continue

                    output_file = os.path.join(output_dir, f"segment_{i:04d}.raw") # Still outputting segment raw files for now
                    segment_raw_files.append(output_file) # Track segment file paths

                    logger.info(f"Generating audio (Context ID: {context_id}) for segment {i}, speaker {speaker} with voice ID {voice_id} and text: {text[:50]}...") # Truncate text for logging
                    await self.generate_audio(text, voice_id, output_file, context_id, continue_stream, language=language) # Pass language to generate_audio

                except Exception as e:
                    logger.error(f"Error generating audio for segment {i}: {str(e)}")
                    continue

            # Concatenate all raw audio segments into a single raw file
            logger.info("Concatenating audio segments...")
            final_raw_file = os.path.join(output_dir, "book_continuous.raw")
            with open(final_raw_file, 'wb') as outfile:
                for raw_file in segment_raw_files:
                    with open(raw_file, 'rb') as infile:
                        outfile.write(infile.read())
            logger.info(f"Concatenated raw audio to: {final_raw_file}")

            # Convert the final raw file to WAV (optional, but for easier playback)
            final_wav_file = os.path.join(output_dir, "book_continuous.wav")
            AudiobookReaderContinuous.convert_raw_to_wav(final_raw_file, final_wav_file) # Use static method for conversion
            logger.info(f"Converted raw audio to WAV: {final_wav_file}")


            metadata_file = os.path.join(output_dir, "metadata.json") # Save metadata - can be optional now
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved metadata to {metadata_file}")


        except Exception as e:
            logger.error(f"Error processing book: {str(e)}")
            raise

    @staticmethod
    def convert_raw_to_wav(raw_filepath, wav_filepath, sample_rate=22050, channels=1, sample_width=4):
        """Static method to convert raw PCM to WAV."""

        with open(raw_filepath, 'rb') as raw_file:
            raw_data = raw_file.read()

        with wave.open(wav_filepath, 'w') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)  # 4 bytes for 32-bit float PCM
            wav_file.setframerate(sample_rate)
            wav_file.setcomptype('NONE', 'not compressed')
            wav_file.writeframes(raw_data)
        logger.info(f"Converted {raw_filepath} to {wav_filepath}")


    def cleanup(self):
        """Cleanup resources before exit - same as before."""
        try:
            del self.gemini_model
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")


if __name__ == "__main__":
    load_dotenv()
    try:
        reader = AudiobookReaderContinuous() # Use the optimized class
        pdf_path = "The Tortoise and the Hare.pdf"
        output_dir = "output_audio_continous" # Different output directory for continuous audio
        logger.info("Starting continuous book processing with Context ID...")
        asyncio.run(reader.process_book(pdf_path, output_dir))
        logger.info("Continuous book processing completed successfully!")
    except Exception as e:
        logger.error(f"Error during continuous execution: {e}")
        raise
    finally:
        if 'reader' in locals():
            reader.cleanup()