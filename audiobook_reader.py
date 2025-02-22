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

voicesUrl = "https://api.cartesia.ai/voices/"

# Configure logging tester
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudiobookReader:
    def __init__(self):
        # Initialize API keys and clients
        self.cartesia_api_key = os.environ.get("CARTESIA_API_KEY")
        self.gemini_api_key = os.environ.get("GOOGLE_API_KEY")
        
        if not self.cartesia_api_key:
            raise ValueError("CARTESIA_API_KEY environment variable is not set")
        if not self.gemini_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
            
        # Initialize API clients
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
            self.voices_prompt += f"ID: {voice['id']}, Description: {voice['description']}\n"
        
        # Register cleanup handler
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

    def analyze_text_with_gemini(self, text: str) -> List[Dict]:
        """
        Use Gemini API to analyze text and identify speakers and sentences.
        Returns a list of dictionaries containing speaker, text, and voice ID information.
        """
        try:
            prompt = f"""
You are provided with a text excerpt from a story. Your task is to process the text as follows:

1. **Sentence Segmentation:**  
   - Break the text into individual sentences using appropriate punctuation (e.g., periods, exclamation points, question marks) as delimiters.  
   - Handle edge cases such as abbreviations, decimal numbers, and titles (e.g., "Dr.", "Mr.", "3.14").  
   - Treat ellipses ("...") and em dashes ("—") carefully, maintaining sentence coherence.

2. **Speaker Identification:**  
   - If a sentence contains dialogue with an explicit speaker name, extract that name.  
   - If the speaker is implied or missing, infer logically based on context or assign "narrator" if unclear.  
   - If a sentence includes multiple dialogue segments with different speakers, split them into separate entries.  
   - Correctly handle nested dialogues or quotes within quotes.

3. **Output Requirements:**  
   - Return a valid JSON array where each element is a JSON object with exactly two fields:  
     - "speaker": the name of the identified speaker (e.g., "narrator", "Alice", etc.)  
     - "text": the sentence text as it appears in the story.  
   - Ensure that the output consists solely of this JSON array, with no additional text, commentary, or markdown formatting.  
   - Escape special characters in the JSON if needed.

Analyze the text below and produce the output accordingly:
{text}
"""

            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Try to parse the JSON
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                # If JSON parsing fails, try to clean up the response
                cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
                try:
                    return json.loads(cleaned_text)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse Gemini response: {response_text}")
                    raise
        except Exception as e:
            logger.error(f"Error in Gemini API call: {str(e)}")
            raise

    
    async def generate_audio(self, text: str, voice_id: str, output_file: str):
        """Generate audio for a piece of text using Cartesia API with WebSocket streaming and HTTP fallback."""
        try:
            # First try the WebSocket approach
            try:
                context_id = str(uuid.uuid4())
                # Updated WebSocket URL to match the API endpoint structure
                ws_url = "wss://api.cartesia.ai/tts/websocket?cartesia_version=2024-06-10&api_key=sk_car_NaJWSjmUaaXoBfSrfOXnX"
                
                headers = {
                    "Cartesia-Version": "2024-06-10",
                    "X-API-Key": self.cartesia_api_key
                }
                
                async with websockets.connect(ws_url) as websocket:
                    # Prepare the initial request
                    request = {
                        "model_id": "sonic",
                        "voice": {
                            "mode": "id",
                            "id": voice_id
                        },
                        "language": "en",
                        "context_id": context_id,
                        "transcript": text[:50],
                        "continue": True,
                        "output_format": {
                            "container": "raw",
                            "encoding": "pcm_s16le",
                            "sample_rate": 8000,
                        }
                    }
                    
                    # Send initial chunk
                    await websocket.send(json.dumps(request))

                    audio_data = bytearray()

                    while True:
                        try:
                            response = await websocket.recv()
                            response_data = json.loads(response)

                            if response_data.get("error"):
                                raise Exception(f"WebSocket error: {response_data['error']}")

                            # Get the data chunk from the response
                            chunk = response_data.get("data")
                            if chunk is not None:
                                if isinstance(chunk, str):
                                    audio_data.extend(chunk.encode("utf-8"))
                                else:
                                    # Assume the chunk is already in a byte-like format
                                    audio_data.extend(chunk)
                            
                            if response_data.get("done", False):
                                break
                        except websockets.exceptions.ConnectionClosed:
                            raise Exception("WebSocket connection closed unexpectedly")


                    # Write the collected audio data to file
                    with open(output_file, "wb") as f:
                        f.write(audio_data)
                    
                    logger.info(f"Generated audio file using WebSocket: {output_file}")
                    return
                    
            except Exception as ws_error:
                logger.warning(f"WebSocket approach failed: {str(ws_error)}, falling back to HTTP API")
                
                # Fallback to regular HTTP API
                try:
                    data = self.cartesia_client.tts.bytes(
                        model_id="sonic",
                        transcript=text,
                        voice_id=voice_id,
                        output_format={
                            "container": "wav",
                            "encoding": "pcm_f32le",
                            "sample_rate": 44100,
                        }
                    )
                    
                    with open(output_file, "wb") as f:
                        f.write(data)
                    logger.info(f"Generated audio file using HTTP API: {output_file}")
                    return
                except Exception as http_error:
                    raise Exception(f"Both WebSocket and HTTP approaches failed. WebSocket error: {ws_error}, HTTP error: {http_error}")
                
        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}")
            raise


    async def process_book(self, pdf_path: str, output_dir: str):
        """Main function to process the book and generate audio files."""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Extract text from PDF
            logger.info("Extracting text from PDF...")
            text = self.extract_text_from_pdf(pdf_path)
            
            # Analyze text and identify speakers with their voice IDs
            logger.info("Analyzing text with Gemini API...")
            analyzed_segments = self.analyze_text_with_gemini(text)
            
                
            # Save metadata
            metadata = {
                "segments": analyzed_segments
            }
            speakers = {}  # Create a dictionary to store unique speakers
            for segment in metadata['segments']:
                speakers[segment['speaker']] = True  # Use speaker as the key and True as the value

            # Add the dictionary to the end of the JSON
            metadata['unique_speakers'] = speakers

            print(metadata['unique_speakers'])

            try:
                # Create the prompt for Gemini to analyze the text and assign voice IDs
                prompt = f"""
                    You are given a JSON file containing two fields: "speaker" and "text". This file provides a sentence-by-sentence transcript of a story. Your task is to analyze the text and assign a voice ID to each speaker based on their personality. Follow these steps:

                    1. **Analyze the Speaker's Personality:**  
                    Examine each speaker's dialogue (or narrative) and infer their personality traits based on their language and tone. Consider characteristics like their emotional state, formal/informal language, and any other relevant aspects of the speech that suggest their persona.

                    2. **Assign Voice IDs Based on Personality:**  
                    Use the available voice descriptions from the list provided to choose an appropriate voice ID for each speaker. Ensure the voice matches the speaker’s inferred personality traits, not randomly assigned.

                    3. **Output Format:**  
                    After processing, return a JSON in the following format:
                    - "speaker1": "voice_id1"
                    - "speaker2": "voice_id2"
                    - Where "speakerX" corresponds to the name of the speaker, and "voice_idX" corresponds to the selected voice ID.

                    Here is the JSON file with speaker information:

                    {metadata}

                    Based on this data, and the list of available voices from the API (described in the following text), assign a voice to each speaker that aligns with their personality. ONLY ADD THE VOICE ID TO THE JSON. NOTHING ELSE!!!!

                    Available voices:
                    {self.voices_prompt}
                    """
                response = self.gemini_model.generate_content(prompt)
                response_text = response.text.strip()

                def clean_json_response(text):
                    # Remove any markdown code block indicators and language specifiers
                    cleaned = text.replace("```python", "").replace("```json", "").replace("```", "").strip()
                    # Remove any leading/trailing whitespace
                    return cleaned.strip()

                try:
                    return_text = json.loads(response_text)
                except json.JSONDecodeError:
                    # First cleaning attempt
                    cleaned_text = clean_json_response(response_text)
                    try:
                        return_text = json.loads(cleaned_text)
                    except json.JSONDecodeError:
                        # Log the actual cleaned text that failed to parse
                        logger.error(f"Failed to parse Gemini response. Original: {response_text}")
                        logger.error(f"Cleaned version: {cleaned_text}")
                        raise

                logger.info(f"Gemini API response: {return_text}")

                # return_text is already a dictionary, no need to parse it again
                voice_id_dict = return_text  # Remove the json.loads() here

                # Add the voice ID dictionary to the metadata
                metadata['unique_speakers'] = voice_id_dict

                # Log the voice IDs for debugging
                logger.info(f"Voice ID dictionary: {metadata['unique_speakers']}")


            except Exception as e:
                logger.error(f"Error in Gemini API call: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Error processing book: {str(e)}")
            raise




        #CODE BELOW IS THE AUDIO OUTPUT GENERATION

        # Generate audio for each segment
        logger.info("Generating audio segments...")
        for i, segment in enumerate(analyzed_segments):
            try:
                text = segment["text"]
                speaker = segment["speaker"]
                # Get the voice ID from the unique_speakers dictionary using the speaker name
                voice_id = metadata['unique_speakers'].get(speaker)
                
                if not voice_id:
                    logger.error(f"No voice ID found for speaker: {speaker}")
                    continue
                    
                output_file = os.path.join(output_dir, f"segment_{i:04d}.raw")
                logger.info(f"Generating audio for speaker {speaker} with voice ID {voice_id} and text {text}")
                await self.generate_audio(text, voice_id, output_file)
            except Exception as e:
                logger.error(f"Error generating audio for segment {i}: {str(e)}")
                # Continue with next segment instead of failing completely
                continue





        metadata_file = os.path.join(output_dir, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_file}")

    def cleanup(self):
        """Cleanup resources before exit"""
        try:
            # Force cleanup of Gemini resources
            del self.gemini_model
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")

# Example usage
if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    
    try:
        # Initialize the reader
        reader = AudiobookReader()
        
        # Define input and output paths
        pdf_path = "The Tortoise and the Hare.pdf"
        output_dir = "output_audio"
        
        # Run the async process
        logger.info("Starting book processing...")
        asyncio.run(reader.process_book(pdf_path, output_dir))
        logger.info("Book processing completed successfully!")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise
    finally:
        # Ensure cleanup is called
        if 'reader' in locals():
            reader.cleanup()