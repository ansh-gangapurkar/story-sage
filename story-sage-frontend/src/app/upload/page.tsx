'use client'

import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { ArrowUpTrayIcon } from '@heroicons/react/24/outline'
import { AudiobookReaderContinuous } from '@/lib/audio_reader'

export default function UploadPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [wavUrl, setWavUrl] = useState<string | null>(null)
  const readerRef = useRef<AudiobookReaderContinuous | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const clientId = useRef<string>(Math.random().toString(36).substring(2))
  const audioContextRef = useRef<AudioContext | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  // Add text input state
  const [inputText, setInputText] = useState<string>('');
  const [inputMethod, setInputMethod] = useState<'text' | 'pdf'>('text');
  const [language, setLanguage] = useState<string>('en');

  useEffect(() => {
    return () => {
      audioContextRef.current?.close();
    };
  }, []);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && file.type === 'application/pdf') {
      setSelectedFile(file)
      setError(null)
    }
  }

  const handleSelectLanguage = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setLanguage(event.target.value);
  };

  const handleCreateAudio = async () => {
    try {
      setIsProcessing(true);
      setError(null);
      
      let file_path: string;
      let output_dir: string;
      
      // Check if we have content to process
      if (inputMethod === 'text' && !inputText.trim()) {
        throw new Error('Please enter some text first');
      }
      
      if (inputMethod === 'pdf' && !selectedFile) {
        throw new Error('Please select a PDF file first');
      }
      
      const formData = new FormData();
      
      if (inputMethod === 'text') {
        // Create a text file from input
        const textBlob = new Blob([inputText], { type: 'text/plain' });
        const textFile = new File([textBlob], 'input.txt');
        formData.append('file', textFile);
      } else {
        // PDF upload
        formData.append('file', selectedFile as File);
      }
      
      // Add selected language to form data
      formData.append('language', language);
      
      // Upload file
      const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.message || 'Failed to process input');
      }
      
      const data = await response.json();
      file_path = data.file_path;
      output_dir = data.output_dir;

      // Create WebSocket connection
      wsRef.current = new WebSocket(`ws://localhost:8000/ws/${clientId.current}`);
      
      // Wait for connection to be established
      await new Promise((resolve, reject) => {
        if (!wsRef.current) return reject('No WebSocket instance');
        
        wsRef.current.onopen = () => {
          resolve(true);
        };
        
        wsRef.current.onerror = (error) => {
          reject(error);
        };
      });

      // Send data after connection is established
      wsRef.current.send(JSON.stringify({
        // Change based on input method
        ...(inputMethod === 'pdf' 
          ? { pdf_path: file_path }  // Send as pdf_path for PDF uploads
          : { text_path: file_path } // Send as text_path for text input
        ),
        output_dir: output_dir,
        language: language
      }));

      wsRef.current.onmessage = async (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'complete' && data.wavUrl) {
          setIsProcessing(false);
          const fullUrl = `http://localhost:8000${data.wavUrl}`;
          setWavUrl(fullUrl);
          
          if (audioRef.current) {
            audioRef.current.src = fullUrl;
            audioRef.current.load();
          }
        } else if (data.type === 'error') {
          setError(data.message);
          setIsProcessing(false);
        }
      };
      
    } catch (err) {
      console.error('Error processing input:', err);
      setError(err instanceof Error ? err.message : 'An error occurred');
      setIsProcessing(false);
    }
  }

  const handlePlayAudio = () => {
    if (audioRef.current) {
      audioRef.current.play();
    }
  }

  const handlePauseAudio = () => {
    if (audioRef.current) {
      audioRef.current.pause();
    }
  }

  const handleDownloadWav = () => {
    if (wavUrl) {
      const link = document.createElement('a');
      link.href = wavUrl;
      link.download = 'audiobook.wav';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  }

  return (
    <main className="min-h-screen relative overflow-hidden">
      {/* Background Pattern */}
      <div className="absolute inset-0 bg-[url('/patterns/natural-paper.png')] opacity-10 z-0" />
      
      {/* Content */}
      <div className="relative z-10">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <h1 className="hero-text mb-6">
              Upload Your Story
            </h1>
            <p className="subtitle-text max-w-2xl mx-auto">
              Transform your written words into a captivating audiobook
            </p>
          </motion.div>

          {/* Input Method Toggle and Content */}
          <div className="mt-8 flex flex-col items-center gap-6">
            {/* Language Selector */}
            <div className="w-full max-w-xl">
              <label className="block mb-4">
                <span className="text-gray-700 font-medium">Select Output Language:</span>
                <select 
                  onChange={handleSelectLanguage} 
                  value={language} 
                  className="block w-full mt-1 p-2 border-2 border-gray-300 rounded-xl focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500"
                >
                  <option value="en">English</option>
                  <option value="es">Spanish</option>
                  <option value="pt">Portuguese</option>
                  <option value="pl">Polish</option>
                  <option value="ru">Russian</option>
                  <option value="ko">Korean</option>
                  <option value="ja">Japanese</option>
                  <option value="hi">Hindi</option>
                  <option value="nl">Dutch</option>
                  <option value="de">German</option>
                  <option value="zh">Chinese</option>
                  <option value="it">Italian</option>
                  <option value="fr">French</option>
                  <option value="tr">Turkish</option>
                  <option value="sv">Swedish</option>
                </select>
              </label>
            </div>

            {/* Input Method Toggle */}
            <div className="flex gap-4">
              <button
                onClick={() => setInputMethod('text')}
                className={`px-4 py-2 rounded ${
                  inputMethod === 'text' 
                    ? 'bg-emerald-500 text-white' 
                    : 'bg-gray-300 text-gray-700'
                }`}
              >
                Enter Text
              </button>
              <button
                onClick={() => setInputMethod('pdf')}
                className={`px-4 py-2 rounded ${
                  inputMethod === 'pdf' 
                    ? 'bg-emerald-500 text-white' 
                    : 'bg-gray-300 text-gray-700'
                }`}
              >
                Upload PDF
              </button>
            </div>

            {/* Text Input */}
            {inputMethod === 'text' && (
              <div className="w-full max-w-xl">
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  placeholder="Paste your story text here..."
                  className="w-full h-48 p-4 border-2 border-gray-400 rounded-xl focus:border-emerald-500 focus:ring-1 focus:ring-emerald-500"
                />
              </div>
            )}

            {/* PDF Uploader */}
            {inputMethod === 'pdf' && (
              <div className="w-full max-w-xl">
                <label
                  className={`relative flex flex-col items-center justify-center w-full h-32 border-2 border-dashed rounded-xl 
                    transition-colors duration-200 ease-in-out cursor-pointer
                    ${selectedFile ? 'border-emerald-500 bg-emerald-50/50' : 'border-gray-400 hover:border-emerald-400'}
                    ${error ? 'border-red-500 bg-red-50/50' : ''}`}
                >
                  <div className="flex flex-col items-center justify-center p-4">
                    <ArrowUpTrayIcon 
                      className={`w-8 h-8 mb-2 ${
                        error ? 'text-red-500' :
                        selectedFile ? 'text-emerald-500' : 'text-gray-400'
                      }`} 
                    />
                    <p className="text-sm text-gray-500">
                      {selectedFile ? (
                        <span className="font-semibold text-emerald-600">{selectedFile.name}</span>
                      ) : (
                        <span>Click to upload PDF or drag and drop</span>
                      )}
                    </p>
                  </div>
                  <input
                    type="file"
                    className="hidden"
                    accept=".pdf"
                    onChange={handleFileChange}
                    disabled={isProcessing}
                  />
                </label>
              </div>
            )}

            {/* Create Audio Button - Show for both methods */}
            {(inputMethod === 'text' ? inputText : selectedFile) && !isProcessing && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-4"
              >
                <button
                  onClick={handleCreateAudio}
                  className="btn-primary"
                >
                  Create Audio
                </button>
              </motion.div>
            )}
          </div>

          {/* Processing Animation - Flower Growing */}
          {isProcessing && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="mt-12 flex flex-col items-center"
            >
              <div className="relative w-24 h-48 flex items-end justify-center">
                {/* Stem */}
                <motion.div
                  className="absolute bottom-0 w-1 bg-green-600"
                  style={{ height: '100px', transformOrigin: 'bottom' }}
                  animate={{ scaleY: [0, 1, 1] }}
                  transition={{
                    duration: 3,
                    times: [0, 0.666, 1],
                    ease: ['easeOut', 'linear'],
                    repeat: Infinity,
                  }}
                />
                {/* Flower */}
                <motion.div
                  className="absolute bottom-[100px] left-1/2 transform -translate-x-1/2"
                >
                  <motion.div
                    className="w-8 h-8 bg-red-500 rounded-full"
                    animate={{
                      scale: [0, 0, 1],
                    }}
                    transition={{
                      duration: 3,
                      times: [0, 0.666, 1],
                      ease: ['easeOut', 'linear'],
                      repeat: Infinity,
                    }}
                  />
                </motion.div>
              </div>
              <p className="mt-6 text-lg text-emerald-700 font-medium">
                {language === 'en' 
                  ? 'Creating your audiobook...'
                  : `Creating your audiobook in ${new Intl.DisplayNames([language], { type: 'language' }).of(language)}...`}
              </p>
            </motion.div>
          )}

          {wavUrl && (
            <div className="mt-8 flex flex-col items-center gap-4">
              {/* Add visible audio element with controls */}
              <audio 
                ref={audioRef}
                controls
                className="w-full max-w-md"
                src={wavUrl}
              />
              
              <div className="flex gap-2">
                <button
                  onClick={handlePlayAudio}
                  className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600"
                >
                  Play
                </button>
                
                <button
                  onClick={handlePauseAudio}
                  className="bg-yellow-500 text-white px-4 py-2 rounded hover:bg-yellow-600"
                >
                  Pause
                </button>
                
                <button
                  onClick={handleDownloadWav}
                  className="bg-purple-500 text-white px-4 py-2 rounded hover:bg-purple-600"
                >
                  Download WAV
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </main>
  )
}