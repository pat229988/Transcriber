from pyannote.audio import Pipeline
import whisperx
import sqlite3
import numpy as np
import sounddevice as sd
from sklearn.metrics.pairwise import cosine_similarity
from pyannote.audio import Model
from pyannote.audio import Inference
import argparse
import torchaudio
import torch
import os
from functools import lru_cache
from collections import defaultdict
from pathlib import Path
import pyaudio
import wave

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
HF_TOKEN = "YOUR_HF_TOKEN_HERE"


def record():
    # Audio recording parameters
    FORMAT = pyaudio.paInt16      # 16-bit resolution
    CHANNELS = 1                  # Mono
    RATE = 44100                  # 44.1kHz sampling rate
    CHUNK = 4096                  # Buffer size
    RECORD_SECONDS = 5            # Duration of recording
    DEVICE_INDEX = 2              # Change this to your mic's device index
    OUTPUT_FILENAME = "output.wav"

    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # List available devices to find your mic's index
    for i in range(audio.get_device_count()):
        print(f"Device {i}: {audio.get_device_info_by_index(i)['name']}")

    # Open stream
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        input_device_index=DEVICE_INDEX,
                        frames_per_buffer=CHUNK)

    print("Recording...")
    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

    print("Finished recording.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded data as a WAV file
    wf = wave.open(OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return os.path.join(os.getcwd, "output.wav")


def init_db():
    db_path = Path('data') / 'speaker_profiles_1.db'
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS speakers
                 (id INTEGER PRIMARY KEY, name TEXT, embedding BLOB)''')
    conn.commit()
    return conn

def store_speaker_profile(db, name, embedding):
    c = db.cursor()
    c.execute("INSERT INTO speakers (name, embedding) VALUES (?, ?)",
              (name, embedding.tobytes()))
    db.commit()

def get_speaker_profiles(db):
    c = db.cursor()
    c.execute("SELECT name, embedding FROM speakers")
    return [(name, np.frombuffer(embedding, dtype=np.float32)) for name, embedding in c.fetchall()]

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

@lru_cache(maxsize=1)
def get_diarization_pipeline():
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                       use_auth_token=HF_TOKEN)
    device = get_device()
    try:
        pipeline.to(device)
    except:
        print(f"Could not move diarization pipeline to {device}. Using default.")
    return pipeline

@lru_cache(maxsize=1)
def get_whisperx_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    try:
        return whisperx.load_model("base.en", device, compute_type=compute_type)
    except ValueError:
        return whisperx.load_model("base.en", device)

@lru_cache(maxsize=1)
def get_embedding_model():
    Model_ = Model.from_pretrained("pyannote/embedding",
                                       use_auth_token=HF_TOKEN)
    device = get_device()
    model = Inference("pyannote/embedding", window="whole", device=device)
    return model

def perform_diarization(audio, sr):
    pipeline = get_diarization_pipeline()
    diarization = pipeline({"waveform": audio, "sample_rate": sr})
    return diarization

def transcribe_audio(audio, sr, diarization):
    model = get_whisperx_model()
    
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy().flatten()
    
    try:
        result = model.transcribe(audio, batch_size=4, language="en")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=model.device)
        result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, model.device, return_char_alignments=False)
    except Exception as e:
        print(f"Transcription error: {e}")
        return []

    merged_turns = merge_diarization_turns(diarization)
    aligned_transcript = align_words_with_speakers(result_aligned["word_segments"], merged_turns)
    
    return aligned_transcript

def merge_diarization_turns(diarization):
    merged = []
    current_turn = None
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if current_turn is None:
            current_turn = {'start': turn.start, 'end': turn.end, 'speaker': speaker}
        else:
            if (speaker == current_turn['speaker'] and 
                turn.start <= current_turn['end'] + 1.0):
                current_turn['end'] = max(current_turn['end'], turn.end)
            else:
                merged.append(current_turn)
                current_turn = {'start': turn.start, 'end': turn.end, 'speaker': speaker}
    if current_turn is not None:
        merged.append(current_turn)
    return merged

def align_words_with_speakers(word_segments, merged_turns):
    aligned_transcript = []
    
    for turn in merged_turns:
        speaker_text = []
        turn_start = turn['start']
        turn_end = turn['end']
        
        for word in word_segments:
            word_start = word.get('start', word.get('start_time', 0))
            word_end = word.get('end', word.get('end_time', 0))
            
            if (word_start >= turn_start and word_end <= turn_end) or \
               (word_start < turn_end and word_end > turn_start):
                speaker_text.append(word['word'])
        
        if speaker_text:
            aligned_transcript.append({
                'start': turn_start,
                'end': turn_end,
                'speaker': turn['speaker'],
                'text': ' '.join(speaker_text).strip()
            })
    
    return aligned_transcript

# Rest of the functions remain the same as previous version

def match_speakers(diarization, db, original_audio):
    matched_speakers = {}
    unknown_speakers = []
    speaker_segments = defaultdict(list)
    
    for turn, _, spk in diarization.itertracks(yield_label=True):
        speaker_segments[spk].append(turn)
    
    profiles = get_speaker_profiles(db)
    
    for spk in speaker_segments:
        segments = speaker_segments[spk]
        if not segments:
            continue
            
        longest_segment = max(segments, key=lambda s: s.end - s.start)
        combined_audio = combine_segments(segments, original_audio)
        if combined_audio is not None:
            embedding = extract_embedding(combined_audio, None)
            best_match, score = find_best_match(embedding, profiles)
            
            if score > 0.85:
                matched_speakers[spk] = best_match
                print(f"Speaker {spk} matched to profile: {best_match} (score: {score:.2f})")
            else:
                unknown_speakers.append((spk, segments, longest_segment))
                print(f"Unknown speaker {spk} detected (best match score: {score:.2f})")
    
    return matched_speakers, unknown_speakers

def combine_segments(segments, original_audio):
    valid_segments = []
    for segment in segments:
        start = int(segment.start * 16000)
        end = int(segment.end * 16000)
        start = max(0, min(start, len(original_audio)-1))
        end = max(0, min(end, len(original_audio)))
        if end > start:
            valid_segments.append(original_audio[start:end])
    
    if not valid_segments:
        return None
    
    return np.concatenate(valid_segments)

def find_best_match(embedding, profiles):
    if not profiles:
        return None, 0

    profile_embeddings = np.vstack([p[1] for p in profiles])
    similarities = cosine_similarity([embedding], profile_embeddings)[0]
    best_idx = np.argmax(similarities)
    return profiles[best_idx][0], similarities[best_idx]

def extract_embedding(audio_input, original_audio=None):
    embedding_model = get_embedding_model()
    device = next(embedding_model.model.parameters()).device
    
    if isinstance(audio_input, (list, np.ndarray)):
        segment_audio = np.array(audio_input).flatten()
    else:
        start_sample = int(audio_input.start * 16000)
        end_sample = int(audio_input.end * 16000)
        segment_audio = original_audio[start_sample:end_sample]
    
    audio_tensor = torch.tensor(segment_audio).unsqueeze(0).to(device)
    return embedding_model({'waveform': audio_tensor, 'sample_rate': 16000})

def tag_unknown_speakers(unknown_speakers, db, original_audio, sr):
    updated_profiles = {}
    for speaker, segments, longest_segment in unknown_speakers:
        tagging_complete = False
        
        while not tagging_complete:
            print(f"\nUnknown speaker (Speaker_{speaker}) detected.")
            print("1: Play audio sample")
            print("2: Enter name for this speaker")
            print("3: Skip this speaker")
            
            choice = input("Enter your choice (1-3): ")
            
            if choice == '1':
                print("Playing longest audio sample...")
                play_audio_snippet(longest_segment, original_audio, sr)
            elif choice == '2':
                combined_audio = combine_segments(segments, original_audio)
                if combined_audio is None:
                    print("Error: No valid audio segments")
                    continue
                    
                name = input(f"Enter a name for Speaker_{speaker}: ")
                if name:
                    embedding = extract_embedding(combined_audio, None)
                    store_speaker_profile(db, name, embedding)
                    updated_profiles[speaker] = name
                    tagging_complete = True
            elif choice == '3':
                tagging_complete = True
                
    return updated_profiles

def play_audio_snippet(speaker_segment, original_audio, sr):
    start_sample = int(speaker_segment.start * sr)
    end_sample = int(speaker_segment.end * sr)
    start_sample = max(0, start_sample)
    end_sample = min(len(original_audio), end_sample)
    
    snippet = original_audio[start_sample:end_sample]
    
    if isinstance(snippet, torch.Tensor):
        snippet = snippet.cpu().numpy()
        
    try:
        sd.play(snippet, sr)
        sd.wait()
    except Exception as e:
        print(f"Error playing audio: {e}")

def export_transcript(transcript, speaker_names, output_file):
    with open(output_file, 'w') as f:
        for entry in transcript:
            speaker = speaker_names.get(entry['speaker'], f"Speaker_{entry['speaker']}")
            f.write(f"[{entry['start']:.2f}-{entry['end']:.2f}] {speaker}: {entry['text']}\n")

def process_audio(file_path, db):
    audio, sr = torchaudio.load(file_path)
    audio = audio.mean(dim=0)  # Convert to mono
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    sr = 16000
    # audio, sr = librosa.load(file_path, sr=16000, mono=True)
    audio_tensor = torch.tensor(audio, device="cpu").unsqueeze(0)
    
    diarization = perform_diarization(audio_tensor, sr)
    matched_speakers, unknown_speakers = match_speakers(diarization, db, audio)
    transcript = transcribe_audio(audio, sr, diarization)
    
    return diarization, transcript, unknown_speakers, audio, sr

def main():
    parser = argparse.ArgumentParser(description="Speaker Diarization and Transcription CLI")
    parser.add_argument("--input", required=False, help="Path to the input audio file")
    parser.add_argument("--output", default="transcript.txt", help="Path to the output transcript file")
    args = parser.parse_args()
    db = init_db()
    if parser.add_argument("--input") :    
        audio_pth = args.input
    else:
        audio_pth = record()

    diarization, transcript, unknown_speakers, audio, sr = process_audio(audio_pth, db)
    updated_profiles = tag_unknown_speakers(unknown_speakers, db, audio, sr)
    
    matched_speakers, _ = match_speakers(diarization, db, audio)
    all_speaker_names = {**matched_speakers, **updated_profiles}
    
    export_transcript(transcript, all_speaker_names, args.output)
    print(f"Transcription saved to {args.output}")

if __name__ == "__main__":
    main()
