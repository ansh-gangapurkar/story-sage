import os
import PyPDF2
import google.generativeai as genai
from cartesia import Cartesia
from typing import Dict, List, Tuple
import json
import asyncio
from dotenv import load_dotenv
import logging
import atexit

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
        self.gemini_model = genai.GenerativeModel('gemini-pro')
        
        # Register cleanup handler
        atexit.register(self.cleanup)
        
        # Default voice IDs for different character types
        self.voice_mappings = {
            "narrator": "b7d50908-b17c-442d-ad8d-810c63997ed9",  # Default narrator voice
            "male_character": "701a96e1-7fdd-4a6c-a81e-a4a450403599",  # Male character voice
            "female_character": "480d702d-0b70-4a32-82c3-93af7b8524ca",  # Female voice
            "child_character": "56b87df1-594d-4135-992c-1112bb504c59",  # Child voice
        }
        
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
        Returns a list of dictionaries containing speaker and text information.
        """
        try:
            prompt = f"""
You are provided with a text excerpt from a story. Your task is to process the text as follows:

1. **Sentence Segmentation:**  
   - Break the text into individual sentences using punctuation (e.g., periods, exclamation points, question marks) as delimiters.

2. **Speaker Identification:**  
   - For each sentence, determine the speaker.  
   - If a sentence contains dialogue with an explicit speaker name, extract that name.  
   - If no explicit speaker is mentioned, assign "narrator" as the speaker.  
   - If a sentence contains multiple dialogue segments with different speakers, split them into separate entries.

3. **Output Requirements:**  
   - Return a valid JSON array where each element is a JSON object with exactly two fields:  
     - "speaker": the name of the identified speaker (e.g., "narrator", "Alice", etc.)  
     - "text": the sentence text as it appears in the story.  
   - Ensure that the output consists solely of this JSON array, with no additional text, commentary, or markdown formatting.

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

    def assign_voice_ids(self, speakers: List[str]) -> Dict[str, str]:
        """Assign Cartesia voice IDs to each unique speaker."""
        voice_assignments = {}
        for speaker in speakers:
            speaker_lower = speaker.lower()
            if speaker_lower == "narrator":
                voice_assignments[speaker] = self.voice_mappings["narrator"]
            elif "mother" in speaker_lower or any(female in speaker_lower for female in ["woman", "girl", "sister"]):
                voice_assignments[speaker] = self.voice_mappings["female_character"]
            elif any(child in speaker_lower for child in ["child", "kid", "little"]):
                voice_assignments[speaker] = self.voice_mappings["child_character"]
            else:
                voice_assignments[speaker] = self.voice_mappings["male_character"]
        return voice_assignments

    async def generate_audio(self, text: str, voice_id: str, output_file: str):
        """Generate audio for a piece of text using Cartesia API."""
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
            logger.info(f"Generated audio file: {output_file}")
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
            
            # Analyze text and identify speakers
            logger.info("Analyzing text with Gemini API...")
            analyzed_segments = self.analyze_text_with_gemini(text)
            
            # Get unique speakers
            unique_speakers = list(set(segment["speaker"] for segment in analyzed_segments))
            logger.info(f"Found speakers: {', '.join(unique_speakers)}")
            
            # Assign voice IDs to speakers
            voice_assignments = self.assign_voice_ids(unique_speakers)
            
            # Generate audio for each segment
            logger.info("Generating audio segments...")
            for i, segment in enumerate(analyzed_segments):
                speaker = segment["speaker"]
                text = segment["text"]
                voice_id = voice_assignments[speaker]
                output_file = os.path.join(output_dir, f"segment_{i:04d}.wav")
                await self.generate_audio(text, voice_id, output_file)
                
            # Save metadata
            metadata = {
                "segments": analyzed_segments,
                "voice_assignments": voice_assignments
            }
            metadata_file = os.path.join(output_dir, "metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved metadata to {metadata_file}")
            
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
        pdf_path = "the-tortoise-and-the-hare-story.pdf"
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