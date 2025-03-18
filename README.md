# Transcriber
transcriber with speaker tagging
# Speaker Diarization and Transcription System

This repository contains a Python-based speaker diarization and transcription system that can:
1. Separate and identify different speakers in an audio recording
2. Transcribe speech to text with speaker attribution
3. Create, store, and match speaker profiles for consistent identification across recordings

## Features

- **Speaker Diarization**: Automatically segments audio by speaker using pyannote.audio
- **Speech Recognition**: Transcribes spoken content with WhisperX
- **Speaker Identification**: Matches speakers against a database of known voices
- **Profile Management**: Creates and stores speaker profiles for future identification
- **Interactive Mode**: Allows listening to and labeling unknown speakers
- **Transcript Export**: Generates timestamped transcripts with speaker identification

## System Workflow

For a visual representation of how the system processes audio files:
- [View the workflow diagram](workflow.md)

## Requirements

- Python 3.8+
- PyTorch
- CUDA-compatible GPU (optional but recommended)
- Hugging Face account and API token

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pat229988/Transcriber.git
   cd Transcriber
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up Hugging Face access:**
   
   The system uses pyannote/speaker-diarization-3.1 which requires a Hugging Face token.
   
   a. Create an account at [Hugging Face](https://huggingface.co/)
   
   b. Generate an access token from your profile settings
   
   c. Accept the license for the pyannote/speaker-diarization-3.1 model at [the model page](https://huggingface.co/pyannote/speaker-diarization-3.1)
   
   d. Replace the token in the code or set it as an environment variable:
   ```python
   # Replace this line in the code
   pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                      use_auth_token="YOUR_HF_TOKEN_HERE")
   ```

   before the first run on windows run the following commands after installing all the required libraries.
   ```bash
   venv\Scripts\activate
   python
   from pyannote.audio import Pipeline
   pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",use_auth_token="YOUR_HF_TOKEN_HERE")
   from pyannote.audio import Model
   model = Model.from_pretrained("pyannote/embedding", use_auth_token="YOUR_ACCESS_TOKEN")
   exit()
   ```


6. **Create a data directory:**
   ```bash
   mkdir data
   ```

## Usage

Run the script with an input audio file:

```bash
python main.py --input path/to/your/audio.wav --output transcript.txt
```

### Command-line Arguments

- `--input`: Path to the input audio file (required)
- `--output`: Path to the output transcript file (default: transcript.txt)

## How It Works

### Key Components

1. **Database Management**: SQLite database for storing speaker profiles
2. **Speaker Embeddings**: Neural embeddings that capture voice characteristics
3. **Diarization Pipeline**: Advanced neural model for speaker segmentation
4. **Speech Recognition**: WhisperX model for accurate transcription
5. **Interactive Interface**: Command-line tools for speaker management

## Technical Details

### Model Information

- **Diarization**: pyannote/speaker-diarization-3.1
- **Transcription**: WhisperX (base.en model)
- **Speaker Embeddings**: pyannote/embedding

### Resource Optimization

- The system uses device detection to utilize GPU when available (CUDA or MPS)
- LRU caching for models to prevent redundant loading
- Efficient processing of audio segments

## License

MIT License

## Acknowledgments

This project relies on several powerful open-source tools:
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarization
- [WhisperX](https://github.com/m-bain/whisperX) for transcription
- [Librosa](https://librosa.org/) for audio processing
