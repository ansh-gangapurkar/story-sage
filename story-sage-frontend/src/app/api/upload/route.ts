import { NextRequest, NextResponse } from 'next/server';
import { writeFile, mkdir } from 'fs/promises';
import path from 'path';
import { AudiobookReaderContinuous } from '@/lib/audio_reader';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    const language = formData.get('language') as string;

    if (!file || !language) {
      return NextResponse.json(
        { error: 'File and language are required' },
        { status: 400 }
      );
    }

    // Create temp directory if it doesn't exist
    const tempDir = path.join(process.cwd(), 'temp');
    await mkdir(tempDir, { recursive: true });

    // Save the uploaded PDF
    const pdfPath = path.join(tempDir, `${Date.now()}_${file.name}`);
    const buffer = Buffer.from(await file.arrayBuffer());
    await writeFile(pdfPath, buffer);

    // Create output directory for this specific upload
    const outputDir = path.join(tempDir, Date.now().toString());
    await mkdir(outputDir, { recursive: true });

    // Initialize the audio reader
    const reader = new AudiobookReaderContinuous();
    
    // Process the book
    await reader.process_book(pdfPath, outputDir);

    // Return the paths to the generated files
    return NextResponse.json({
      success: true,
      outputDir: outputDir,
    });

  } catch (error) {
    console.error('Error processing upload:', error);
    return NextResponse.json(
      { error: 'Error processing the file' },
      { status: 500 }
    );
  }
} 