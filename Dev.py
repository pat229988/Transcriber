from pyannote.audio import Pipeline, Inference
import whisperx
import sqlite3
import numpy as np
import sounddevice as sd
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import torchaudio
import torch
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

class SpeakerDatabase:
    """Handles speaker profile storage and retrieval"""
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database schema"""
        self.conn = sqlite3.connect(self.db_path)
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS speakers
                     (id INTEGER PRIMARY KEY, name TEXT, embedding BLOB)''')
        self.conn.commit()

    def store_profile(self, name: str, embedding: np.ndarray) -> int:
        """Store a new speaker profile"""
        c = self.conn.cursor()
        c.execute("INSERT INTO speakers (name, embedding) VALUES (?, ?)",
                  (name, embedding.tobytes()))
        self.conn.commit()
        return c.lastrowid

    def get_profiles(self) -> List[Tuple[str, np.ndarray]]:
        """Retrieve all stored profiles"""
        c = self.conn.cursor()
        c.execute("SELECT name, embedding FROM speakers")
        return [(name, np.frombuffer(embedding, dtype=np.float32)) 
                for name, embedding in c.fetchall()]

class AudioProcessor:
    """Handles audio processing and playback operations"""
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.device = self._get_audio_device()

    @staticmethod
    def _get_audio_device() -> torch.device:
        """Determine best available compute device"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load_audio(self, file_path: Path) -> Tuple[torch.Tensor, int]:
        """Load and preprocess audio file"""
        audio, sr = torchaudio.load(file_path)
        audio = audio.mean(dim=0)  # Convert to mono
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
        return audio, self.sample_rate

    def play_segment(self, audio: np.ndarray, max_duration: float = 10.0):
        """Play audio segment with duration limit"""
        max_samples = int(max_duration * self.sample_rate)
        playable = audio[:max_samples] if len(audio) > max_samples else audio
        try:
            sd.play(playable, self.sample_rate)
            sd.wait()
        except Exception as e:
            print(f"Audio playback error: {e}")

class ModelLoader:
    """Manages model loading and configuration"""
    def __init__(self, model_dir: Path, device: torch.device):
        self.model_dir = model_dir
        self.device = device
        self.whisperx_device = self._get_whisperx_device()
        self._load_models()
        self.auth_token = "YOUR_HF_TOKEN_HERE"

    def _get_whisperx_device(self) -> str:
        """Determine appropriate device for WhisperX"""
        return "cpu" if self.device.type == "mps" else self.device.type

    def _load_models(self):
        """Load all required models"""
        self.diarization_pipeline = self._load_diarization_model()
        self.whisper_model, self.embedding_model = self._load_remaining_models()

    def _load_diarization_model(self) -> Pipeline:
        """Load speaker diarization pipeline"""
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.auth_token
            ).to(self.device)
        except Exception as e:
            print(f"Diarization model loading warning: {e}")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.auth_token
            )
        return pipeline

    def _load_remaining_models(self) -> Tuple[Any, Any]:
        """Load WhisperX and embedding models"""
        compute_type = "float16" if self.whisperx_device == "cuda" else "int8"
        whisper_model = whisperx.load_model(
            "base.en", self.whisperx_device, compute_type=compute_type
        )
        embedding_model = Inference(
            "pyannote/embedding",
            use_auth_token=self.auth_token,
            window="whole",
            device=self.device
        )
        return whisper_model, embedding_model

class DiarizationProcessor:
    """Handles core diarization and transcription logic"""
    def __init__(self, model_loader: ModelLoader, audio_processor: AudioProcessor):
        self.models = model_loader
        self.audio_processor = audio_processor
        self.whisper_align_model = None

    def process_audio(self, audio_path: Path) -> dict:
        """Main processing pipeline for audio files"""
        audio, sr = self.audio_processor.load_audio(audio_path)
        diarization = self._perform_diarization(audio)
        transcript = self._transcribe_audio(audio.numpy(), sr, diarization)
        return {
            "diarization": diarization,
            "transcript": transcript,
            "audio": audio.numpy(),
            "sr": sr
        }

    def _perform_diarization(self, audio: torch.Tensor) -> Pipeline:
        """Run speaker diarization on audio"""
        return self.models.diarization_pipeline({
            "waveform": audio.unsqueeze(0).to(self.models.device),
            "sample_rate": self.audio_processor.sample_rate
        })

    def _transcribe_audio(self, audio: np.ndarray, sr: int, diarization: Pipeline) -> List[dict]:
        result = self.models.whisper_model.transcribe(audio, batch_size=4, language="en")
        aligned = self._align_transcription(result, audio)
        print("Aligned structure:", aligned.keys())  # Debug print
        if "word_segments" in aligned:
            print("Word segment structure:", aligned["word_segments"][0] if aligned["word_segments"] else "Empty")
        return self._merge_transcript_with_speakers(aligned, diarization)


    def _align_transcription(self, result: dict, audio: np.ndarray) -> dict:
        """Align Whisper transcription with audio"""
        model_a, metadata = self._get_align_model(result["language"])
        return whisperx.align(
            result["segments"], model_a, metadata, audio, 
            self.models.whisperx_device, return_char_alignments=False
        )

    def _get_align_model(self, language: str) -> Tuple[Any, dict]:
        """Load or retrieve alignment model"""
        if self.whisper_align_model is None:
            model_a, metadata = whisperx.load_align_model(
                language_code=language, device=self.models.whisperx_device
            )
            self.whisper_align_model = (model_a, metadata)
        return self.whisper_align_model

    def _merge_transcript_with_speakers(self, aligned: dict, diarization: Pipeline) -> List[dict]:
        """Merge transcription with diarization results"""
        merged_turns = self._merge_diarization_turns(diarization)
        return self._align_words_with_speakers(aligned["word_segments"], merged_turns)

    def _merge_diarization_turns(self, diarization: Pipeline) -> List[dict]:
        """Merge consecutive speaker turns"""
        merged = []
        current_turn = None
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if not current_turn:
                current_turn = {'start': turn.start, 'end': turn.end, 'speaker': speaker}
            else:
                if speaker == current_turn['speaker'] and turn.start <= current_turn['end'] + 1.0:
                    current_turn['end'] = max(current_turn['end'], turn.end)
                else:
                    merged.append(current_turn)
                    current_turn = {'start': turn.start, 'end': turn.end, 'speaker': speaker}
        if current_turn:
            merged.append(current_turn)
        return merged

    def _align_words_with_speakers(self, words: List[dict], turns: List[dict]) -> List[dict]:
        transcript = []
        for turn in turns:
            speaker_text = []
            for word in words:
                word_start = word.get('start') or word.get('timestamp', [0])[0]
                word_end = word.get('end') or word.get('timestamp', [0])[-1]
                if (word_start >= turn['start'] and word_end <= turn['end']) or \
                (word_start < turn['end'] and word_end > turn['start']):
                    speaker_text.append(word.get('word', word.get('text', '')))
            if speaker_text:
                transcript.append({
                    'start': turn['start'],
                    'end': turn['end'],
                    'speaker': turn['speaker'],
                    'text': ' '.join(speaker_text).strip()
                })
        return transcript


class SpeakerIdentifier:
    """Handles speaker identification and matching"""
    def __init__(self, db: SpeakerDatabase, embedding_model: Inference):
        self.db = db
        self.embedding_model = embedding_model
        self.similarity_threshold = 0.85

    def match_speakers(self, diarization: Pipeline, audio: np.ndarray) -> Tuple[dict, list]:
        """Match diarized speakers to known profiles"""
        speaker_segments = self._segment_by_speaker(diarization)
        profiles = self.db.get_profiles()
        matched, unknown = {}, []

        for spk, segments in speaker_segments.items():
            embedding = self._get_speaker_embedding(segments, audio)
            if embedding is None:
                continue

            match, score = self._find_profile_match(embedding, profiles)
            if score >= self.similarity_threshold:
                matched[spk] = match
            else:
                unknown.append((spk, segments))
        return matched, unknown

    def _segment_by_speaker(self, diarization: Pipeline) -> Dict[str, list]:
        """Group diarization segments by speaker"""
        speaker_segments = defaultdict(list)
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments[speaker].append(turn)
        return speaker_segments

    def _get_speaker_embedding(self, segments: list, audio: np.ndarray) -> Optional[np.ndarray]:
        """Generate embedding from speaker segments"""
        combined = self._combine_segments(segments, audio)
        if combined is None:
            return None
        return self.embedding_model({
            "waveform": torch.tensor(combined).unsqueeze(0), 
            "sample_rate": 16000
        })

    def _combine_segments(self, segments: list, audio: np.ndarray) -> Optional[np.ndarray]:
        """Combine audio segments for embedding generation"""
        valid_segments = []
        for seg in segments:
            start = int(seg.start * 16000)
            end = int(seg.end * 16000)
            start = max(0, min(start, len(audio)-1))
            end = max(0, min(end, len(audio)))
            if end > start:
                valid_segments.append(audio[start:end])
        return np.concatenate(valid_segments) if valid_segments else None

    def _find_profile_match(self, embedding: np.ndarray, profiles: List[Tuple[str, np.ndarray]]) -> Tuple[Optional[str], float]:
        """Find best matching profile for embedding"""
        if not profiles:
            return None, 0.0
        profile_embeds = np.array([p[1] for p in profiles])
        similarities = cosine_similarity([embedding], profile_embeds)[0]
        best_idx = np.argmax(similarities)
        return profiles[best_idx][0], similarities[best_idx]

class InteractiveTagger:
    """Handles interactive tagging of unknown speakers"""
    def __init__(self, audio_processor: AudioProcessor, db: SpeakerDatabase, embedding_model: Inference):
        self.audio_processor = audio_processor
        self.db = db
        self.embedding_model = embedding_model
        self.target_duration = 10.0  # Seconds for playback

    def tag_unknowns(self, unknown_speakers: list, audio: np.ndarray) -> dict:
        """Interactive tagging process for unknown speakers"""
        updated = {}
        for spk, segments in unknown_speakers:
            print(f"\nProcessing unknown speaker: {spk}")
            sample = self._create_playback_sample(segments, audio)
            if sample is not None and self._present_options(spk, sample):
                name = self._register_new_speaker(spk, segments, audio)
                if name:
                    updated[spk] = name
        return updated

    def _create_playback_sample(self, segments: list, audio: np.ndarray) -> Optional[np.ndarray]:
        """Create optimal playback sample up to target duration"""
        combined = []
        duration = 0.0
        for seg in sorted(segments, key=lambda x: x.start):
            if duration >= self.target_duration:
                break
            seg_duration = seg.end - seg.start
            if duration + seg_duration > self.target_duration:
                partial_duration = self.target_duration - duration
                partial_seg = seg.extract(audio)
                partial_seg = partial_seg[:int(partial_duration * 16000)]
                combined.append(partial_seg)
                break
            combined.append(seg.extract(audio))
            duration += seg_duration
        return np.concatenate(combined) if combined else None

    def _present_options(self, spk: str, sample: np.ndarray) -> bool:
        """Present interactive tagging options"""
        self.audio_processor.play_segment(sample)
        choice = input(f"Tag {spk}? (y/n): ")
        return choice.lower() == 'y'

    def _register_new_speaker(self, spk: str, segments: list, audio: np.ndarray) -> Optional[str]:
        """Register new speaker profile using all available segments"""
        name = input(f"Enter name for {spk}: ").strip()
        combined = self._combine_all_segments(segments, audio)
        
        if combined is None:
            print("Error: No valid audio segments for registration")
            return None
            
        duration = len(combined) / 16000
        if duration < 5:
            print(f"Warning: Registration audio is short ({duration:.1f}s). For better accuracy, collect at least 5s of speech.")
        
        embedding = self.embedding_model({
            "waveform": torch.tensor(combined).unsqueeze(0),
            "sample_rate": 16000
        })
        self.db.store_profile(name, embedding)
        return name

    def _combine_all_segments(self, segments: list, audio: np.ndarray) -> Optional[np.ndarray]:
        """Combine all available segments for registration"""
        valid_segments = []
        for seg in segments:
            start = int(seg.start * 16000)
            end = int(seg.end * 16000)
            start = max(0, min(start, len(audio)-1))
            end = max(0, min(end, len(audio)))
            if end > start:
                valid_segments.append(audio[start:end])
        return np.concatenate(valid_segments) if valid_segments else None

class SpeakerDiarizationApp:
    """Main application class coordinating all components"""
    def __init__(self, model_dir: Path = Path("models"), db_path: Path = Path("data/speaker_profiles.db")):
        self.model_dir = model_dir
        self.db = SpeakerDatabase(db_path)
        self.audio_processor = AudioProcessor()
        self.model_loader = ModelLoader(model_dir, self.audio_processor.device)
        self.diarization_processor = DiarizationProcessor(self.model_loader, self.audio_processor)
        self.speaker_identifier = SpeakerIdentifier(self.db, self.model_loader.embedding_model)
        self.tagger = InteractiveTagger(self.audio_processor, self.db, self.model_loader.embedding_model)

    def process_file(self, input_path: Path, output_path: Path):
        """Process audio file end-to-end"""
        results = self.diarization_processor.process_audio(input_path)
        matched, unknown = self.speaker_identifier.match_speakers(results["diarization"], results["audio"])
        updated = self.tagger.tag_unknowns(unknown, results["audio"])
        self._export_transcript(results["transcript"], {**matched, **updated}, output_path)

    def _export_transcript(self, transcript: list, speakers: dict, output_path: Path):
        """Export final transcript with speaker names"""
        with open(output_path, 'w') as f:
            for entry in transcript:
                speaker = speakers.get(entry['speaker'], f"Speaker_{entry['speaker']}")
                f.write(f"[{entry['start']:.2f}-{entry['end']:.2f}] {speaker}: {entry['text']}\n")

def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(description="Optimized Speaker Diarization System")
    parser.add_argument("--input", required=True, type=Path, help="Input audio file path")
    parser.add_argument("--output", type=Path, default="transcript.txt", help="Output transcript path")
    args = parser.parse_args()

    app = SpeakerDiarizationApp()
    app.process_file(args.input, args.output)
    print(f"Processing complete. Transcript saved to {args.output}")

if __name__ == "__main__":
    main()