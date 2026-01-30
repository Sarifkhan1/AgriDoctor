"""
AgriDoctor AI - Speech to Text Pipeline
Transcribe farmer voice notes using ASR with timestamps and confidence.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """A segment of transcribed speech."""
    start_sec: float
    end_sec: float
    text: str
    confidence: float


@dataclass
class Transcript:
    """Complete transcript with metadata."""
    encounter_id: str
    media_id: str
    language: str
    language_confidence: float
    full_text: str
    duration_sec: float
    segments: List[TranscriptSegment]
    model_name: str
    

class ASRTranscriber:
    """Speech-to-text transcription for farmer voice notes."""
    
    def __init__(
        self,
        model_name: str = "openai/whisper-base",
        device: str = "auto",
        language: Optional[str] = None
    ):
        """
        Initialize ASR transcriber.
        
        Args:
            model_name: Whisper model name (tiny, base, small, medium, large)
            device: 'cuda', 'cpu', or 'auto'
            language: Force specific language or None for auto-detect
        """
        self.model_name = model_name
        self.language = language
        self.device = self._get_device(device)
        self.model = None
        self.processor = None
        self._whisper_model = None
        
    def _get_device(self, device: str) -> str:
        """Determine the best available device."""
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device
    
    def _load_whisper(self):
        """Load Whisper model (lazy loading)."""
        if self._whisper_model is not None:
            return
            
        try:
            import whisper
            
            # Map model name to whisper model
            model_size = self.model_name.replace("openai/whisper-", "")
            if model_size not in ["tiny", "base", "small", "medium", "large"]:
                model_size = "base"
            
            logger.info(f"Loading Whisper {model_size} on {self.device}...")
            self._whisper_model = whisper.load_model(model_size, device=self.device)
            logger.info("Whisper model loaded successfully")
            
        except ImportError:
            logger.warning("Whisper not installed. Trying transformers pipeline...")
            self._load_transformers_asr()
    
    def _load_transformers_asr(self):
        """Load transformers-based ASR as fallback."""
        try:
            from transformers import pipeline
            import torch
            
            logger.info("Loading transformers ASR pipeline...")
            
            device_num = 0 if self.device == "cuda" else -1
            
            self.processor = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-base",
                device=device_num,
                chunk_length_s=30,
                return_timestamps=True
            )
            logger.info("Transformers ASR pipeline loaded")
            
        except Exception as e:
            logger.error(f"Failed to load ASR model: {e}")
            raise RuntimeError("No ASR backend available. Install whisper or transformers.")
    
    def transcribe(
        self,
        audio_path: str,
        encounter_id: str = "",
        media_id: str = ""
    ) -> Transcript:
        """
        Transcribe an audio file.
        
        Args:
            audio_path: Path to audio file (wav, mp3, etc.)
            encounter_id: Encounter identifier
            media_id: Media file identifier
            
        Returns:
            Transcript object with segments and metadata
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Ensure model is loaded
        if self._whisper_model is None and self.processor is None:
            self._load_whisper()
        
        # Transcribe based on available backend
        if self._whisper_model is not None:
            return self._transcribe_whisper(audio_path, encounter_id, media_id)
        else:
            return self._transcribe_transformers(audio_path, encounter_id, media_id)
    
    def _transcribe_whisper(
        self,
        audio_path: Path,
        encounter_id: str,
        media_id: str
    ) -> Transcript:
        """Transcribe using OpenAI Whisper."""
        import whisper
        
        logger.info(f"Transcribing: {audio_path}")
        
        # Transcribe with word-level timestamps
        result = self._whisper_model.transcribe(
            str(audio_path),
            language=self.language,
            word_timestamps=True,
            verbose=False
        )
        
        # Extract segments
        segments = []
        for seg in result.get("segments", []):
            segments.append(TranscriptSegment(
                start_sec=round(seg["start"], 3),
                end_sec=round(seg["end"], 3),
                text=seg["text"].strip(),
                confidence=round(seg.get("avg_logprob", -1.0) * -0.1 + 1.0, 3)  # Convert log prob to confidence
            ))
        
        # Get duration
        duration = segments[-1].end_sec if segments else 0.0
        
        # Language detection confidence
        language = result.get("language", "en")
        lang_probs = result.get("language_probs", {})
        lang_confidence = lang_probs.get(language, 0.9)
        
        return Transcript(
            encounter_id=encounter_id or audio_path.stem,
            media_id=media_id or audio_path.stem,
            language=language,
            language_confidence=round(lang_confidence, 3),
            full_text=result.get("text", "").strip(),
            duration_sec=round(duration, 3),
            segments=segments,
            model_name=self.model_name
        )
    
    def _transcribe_transformers(
        self,
        audio_path: Path,
        encounter_id: str,
        media_id: str
    ) -> Transcript:
        """Transcribe using transformers pipeline."""
        logger.info(f"Transcribing (transformers): {audio_path}")
        
        result = self.processor(str(audio_path), return_timestamps=True)
        
        # Extract segments from chunks
        segments = []
        if "chunks" in result:
            for chunk in result["chunks"]:
                timestamps = chunk.get("timestamp", (0, 0))
                segments.append(TranscriptSegment(
                    start_sec=round(timestamps[0] or 0, 3),
                    end_sec=round(timestamps[1] or 0, 3),
                    text=chunk.get("text", "").strip(),
                    confidence=0.8  # Default confidence for pipeline
                ))
        
        duration = segments[-1].end_sec if segments else 0.0
        
        return Transcript(
            encounter_id=encounter_id or audio_path.stem,
            media_id=media_id or audio_path.stem,
            language="en",  # Pipeline doesn't easily expose language
            language_confidence=0.8,
            full_text=result.get("text", "").strip(),
            duration_sec=round(duration, 3),
            segments=segments,
            model_name="transformers-whisper"
        )
    
    def save_transcript(
        self,
        transcript: Transcript,
        output_path: str
    ) -> Path:
        """
        Save transcript to JSON file.
        
        Args:
            transcript: Transcript object
            output_path: Output file path or directory
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        
        # If directory, create filename
        if output_path.is_dir():
            output_path = output_path / f"{transcript.media_id}_transcript.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict
        data = {
            "encounter_id": transcript.encounter_id,
            "media_id": transcript.media_id,
            "language": transcript.language,
            "language_confidence": transcript.language_confidence,
            "full_text": transcript.full_text,
            "duration_sec": transcript.duration_sec,
            "model_name": transcript.model_name,
            "segments": [asdict(seg) for seg in transcript.segments]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved transcript: {output_path}")
        return output_path
    
    def transcribe_directory(
        self,
        input_dir: str,
        output_dir: str,
        extensions: List[str] = None
    ) -> List[Path]:
        """
        Transcribe all audio files in a directory.
        
        Args:
            input_dir: Directory with audio files
            output_dir: Directory for output transcripts
            extensions: Audio file extensions
            
        Returns:
            List of output transcript paths
        """
        if extensions is None:
            extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find audio files
        audio_files = []
        for ext in extensions:
            audio_files.extend(input_path.rglob(f"*{ext}"))
            audio_files.extend(input_path.rglob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(audio_files)} audio files")
        
        outputs = []
        errors = []
        
        for i, audio_file in enumerate(audio_files):
            try:
                # Extract encounter_id from filename if possible
                parts = audio_file.stem.split("_")
                encounter_id = parts[0] if len(parts) > 1 else audio_file.stem
                
                transcript = self.transcribe(
                    str(audio_file),
                    encounter_id=encounter_id,
                    media_id=audio_file.stem
                )
                
                out_file = self.save_transcript(transcript, output_path)
                outputs.append(out_file)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(audio_files)} files")
                    
            except Exception as e:
                logger.error(f"Error transcribing {audio_file}: {e}")
                errors.append((str(audio_file), str(e)))
        
        logger.info(f"Successfully transcribed {len(outputs)} files")
        if errors:
            logger.warning(f"Failed to transcribe {len(errors)} files")
        
        return outputs


def main():
    """CLI for ASR transcription."""
    parser = argparse.ArgumentParser(description="Transcribe farmer voice notes")
    parser.add_argument("--input", "-i", required=True, help="Input audio file or directory")
    parser.add_argument("--output", "-o", default="./transcripts", help="Output directory")
    parser.add_argument("--model", "-m", default="base", 
                       choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size")
    parser.add_argument("--language", "-l", default=None, help="Force language (e.g., 'en', 'hi')")
    parser.add_argument("--device", "-d", default="auto", choices=["auto", "cuda", "cpu"])
    
    args = parser.parse_args()
    
    # Initialize transcriber
    transcriber = ASRTranscriber(
        model_name=f"openai/whisper-{args.model}",
        device=args.device,
        language=args.language
    )
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file
        transcript = transcriber.transcribe(str(input_path))
        out_path = transcriber.save_transcript(transcript, args.output)
        print(f"\nTranscript saved: {out_path}")
        print(f"Language: {transcript.language} (conf: {transcript.language_confidence})")
        print(f"Duration: {transcript.duration_sec}s")
        print(f"\nText:\n{transcript.full_text}")
        
    elif input_path.is_dir():
        # Directory
        outputs = transcriber.transcribe_directory(str(input_path), args.output)
        print(f"\nTranscribed {len(outputs)} files to {args.output}")
    
    else:
        print(f"Error: {args.input} does not exist")
        sys.exit(1)


if __name__ == "__main__":
    main()
