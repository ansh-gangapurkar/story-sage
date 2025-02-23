// The browser's WebSocket API will be used automatically

interface AudioSegment {
  speaker: string;
  text: string;
  voice_id: string;
}

interface Metadata {
  segments: AudioSegment[];
  unique_speakers: Record<string, string>;
}

interface AudioMessage {
  type: 'audio';
  audio: string;
}

interface MetadataMessage {
  type: 'metadata';
  metadata: Metadata;
}

interface WebSocketMessage {
  type: 'audio' | 'complete' | 'error';
  audio?: string;
  message?: string;
  wavUrl?: string;  // Add this for the WAV file URL
}

export class AudiobookReaderContinuous {
  private audioElement: HTMLAudioElement | null = null;
  private ws: WebSocket | null = null;
  private isProcessing: boolean = false;

  constructor() {
    this.audioElement = new Audio();
  }

  async process_book(pdfPath: string, outputDir: string, language: string = 'en'): Promise<void> {
    try {
      this.isProcessing = true;
      this.ws = new WebSocket('ws://localhost:8000/ws/process');

      this.ws.send(JSON.stringify({
        pdf_path: pdfPath,
        output_dir: outputDir,
        language: language
      }));

      this.ws.onmessage = async (event: MessageEvent) => {
        const data = JSON.parse(event.data) as WebSocketMessage;
        
        if (data.type === 'complete' && data.wavUrl) {
          this.isProcessing = false;
          // Set audio source to the WAV file URL
          if (this.audioElement) {
            this.audioElement.src = data.wavUrl;
          }
        }
      };
    } catch (error) {
      this.isProcessing = false;
      throw error;
    }
  }

  playAudio() {
    this.audioElement?.play();
  }

  pauseAudio() {
    this.audioElement?.pause();
  }

  cleanup(): void {
    if (this.ws) {
      this.ws.close();
    }
    if (this.audioElement) {
      this.audioElement.pause();
      this.audioElement = null;
    }
  }
}