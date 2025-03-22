# faster_whisper_transcriber.py
from langchain_community.document_loaders.blob_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.generic import GenericLoader
import tempfile
import os
from pathlib import Path
import yt_dlp
from faster_whisper import WhisperModel
from typing import Iterator, List

class FasterWhisperParser:
    """Parser for audio files using faster-whisper."""
    
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        """
        Initialize the faster-whisper parser.
        
        Args:
            model_size: Size of the model to use (tiny, base, small, medium, large-v1, large-v2, large-v3)
            device: Device to use for inference (cpu or cuda)
            compute_type: Compute type (int8, int16, float16, float32)
        """
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print(f"Loaded faster-whisper model: {model_size} on {device} using {compute_type}")
    
    def parse(self, blob):
        """
        Parse audio file using faster-whisper.
        
        Args:
            blob: File blob to parse
        
        Returns:
            List of documents with the transcription
        """
        from langchain.schema.document import Document
        
        # Create a temporary file to save the audio content
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_file.write(blob.as_bytes())
            temp_path = temp_file.name
        
        try:
            # Transcribe using faster-whisper
            segments, info = self.model.transcribe(temp_path, beam_size=5)
            
            # Combine all segments into a single transcript
            transcript = ""
            for segment in segments:
                transcript += segment.text + " "
            
            # Create document with metadata
            metadata = {
                "source": blob.source,
                "language": info.language,
                "language_probability": info.language_probability
            }
            
            return [Document(page_content=transcript.strip(), metadata=metadata)]
        
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
    
    def lazy_parse(self, blob) -> Iterator[List]:
        """
        Lazily parse the blob - required by GenericLoader.
        
        Args:
            blob: File blob to parse
            
        Yields:
            List of documents
        """
        yield self.parse(blob)

def download_youtube_audio(urls, output_dir):
    """
    Download audio from YouTube videos using yt-dlp directly.
    
    Args:
        urls: List of YouTube URLs
        output_dir: Directory to save the audio files
    
    Returns:
        List of downloaded audio file paths
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    downloaded_files = []
    
    for url in urls:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': f'{output_dir}/%(id)s.%(ext)s',
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info).replace('.webm', '.mp3').replace('.m4a', '.mp3')
            downloaded_files.append(filename)
    
    return downloaded_files

def transcribe_youtube_videos(urls, output_dir="./youtube_audio", model_size="base"):
    """
    Transcribe YouTube videos using faster-whisper.
    
    Args:
        urls: List of YouTube URLs
        output_dir: Directory to save the audio files
        model_size: Size of the whisper model to use
    
    Returns:
        List of documents with transcriptions
    """
    # First, download audio from YouTube videos using yt-dlp directly
    print("Downloading YouTube audio...")
    download_youtube_audio(urls, output_dir)
    print(f"Audio downloaded to {output_dir}")
    
    # Set up the parser
    whisper_parser = FasterWhisperParser(model_size=model_size)
    
    # Use GenericLoader to combine the blob loader and parser
    print("Transcribing audio files...")
    blob_loader = FileSystemBlobLoader(output_dir)
    loader = GenericLoader(blob_loader, whisper_parser)
    
    # This will load the audio files and transcribe them
    documents = loader.load()
    print(f"Transcription complete. Generated {len(documents)} documents.")
    
    return documents

# # For direct usage
# if __name__ == "__main__":
#     url = ["https://www.youtube.com/watch?v=jGwO_UgTS7I"]
#     save_dir = "docs/youtube/"
#     docs = transcribe_youtube_videos(url, save_dir, model_size="base")
#     print(f"Transcribed {len(docs)} documents")
#     if docs:
#         print("First document content preview:")
#         print(docs[0].page_content[:500] + "...")
#         print(f"Detected language: {docs[0].metadata['language']} (confidence: {docs[0].metadata['language_probability']:.2f})")