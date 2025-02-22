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

voicesUrl = "https://api.cartesia.ai/voices/"

# Configure logging
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


    #CODE BELOW IS THE FUNCTION FOR AUDIO GENERATION
    # async def generate_audio(self, text: str, voice_id: str, output_file: str):
    #     """Generate audio for a piece of text using Cartesia API."""
    #     try:
    #         data = self.cartesia_client.tts.bytes(
    #             model_id="sonic",
    #             transcript=text,
    #             voice_id=voice_id,
    #             output_format={
    #                 "container": "wav",
    #                 "encoding": "pcm_f32le",
    #                 "sample_rate": 44100,
    #             }
    #         )
            
    #         with open(output_file, "wb") as f:
    #             f.write(data)
    #         logger.info(f"Generated audio file: {output_file}")
    #     except Exception as e:
    #         logger.error(f"Error generating audio: {str(e)}")
    #         raise

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
            

            #CODE BELOW IS THE AUDIO OUTPUT GENERATION

            # Generate audio for each segment
            # logger.info("Generating audio segments...")
            # for i, segment in enumerate(analyzed_segments):
            #     print(segment)
            #     text = segment["text"]
            #     voice_id = segment["id"]
            #     output_file = os.path.join(output_dir, f"segment_{i:04d}.wav")
            #     await self.generate_audio(text, voice_id, output_file)
                
            # Save metadata
            metadata = {
                "segments": analyzed_segments
            }
            speakers = {}  # Create a dictionary to store unique speakers
            for segment in metadata['segments']:
                speakers[segment['speaker']] = True  # Use speaker as the key and True as the value

            # Add the dictionary to the end of the JSON
            metadata['unique_speakers'] = speakers

            print(metadata)








            try:
                # Create the prompt for Gemini to analyze the text and assign voice IDs
                prompt = f"""
            You are provided with a json file with two fields, speaker and text, which is a per sentence transcript of a story. Your task is to process the text as follows:
            go through all of the lines and understand the personality of each speaker based on their text.
                - "speaker": the name of the identified speaker (e.g., "narrator", "Alice", etc.)  
                - "text": the sentence text as it appears in the story.  

            based on these available voices, in 
            {self.voices_prompt} which includes all the available voice ids, and their description, assign the voice id to the speaker based on their personality.
            Make sure to assign the voice id to the speaker based on their personality and not randomly.

            Return a dictionary in the following format:
                "speaker1": "voice_id1",
                "speaker2": "voice_id2",
            """

                # Call the Gemini API to generate the response based on the prompt
                response = self.gemini_model.generate_content(prompt)
                response_text = response.text.strip()
                
                # Log or print the response text to check what we're getting
                logger.info(f"Gemini API response: {response_text}")
                
                # Attempt to parse the response if it's valid JSON
                voice_id_dict = json.loads(response_text)

                # Update metadata with the voice ID dictionary
                metadata['voice_ids'] = voice_id_dict  # Add the new dictionary under the key "voice_ids"

                # Optionally, print the voice ID dictionary to check the result
                print("Voice ID Dictionary:", voice_id_dict)

            except Exception as e:
                logger.error(f"Error in Gemini API call: {str(e)}")
                raise
















            # Save the updated metadata to the file
            metadata_file = os.path.join(output_dir, "metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved updated metadata with voice IDs to {metadata_file}")

        
            
        except Exception as e:
            logger.error(f"Error processing book: {str(e)}")
            raise

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
