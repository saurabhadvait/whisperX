import json
import os
# import warnings; warnings.filterwarnings("ignore")
# Define paths and parameters
import sys

import numpy as np
import torch
from pydub import AudioSegment

from src.prev_transcribe import \
    transcribe_audio as transcribe_audio_with_openai
from src.prev_transcribe import write
from whisperx.asr import (SAMPLE_RATE, List, SingleSegment, load_audio,
                          load_vad_model, merge_chunks)

audio_path = sys.argv[1]
output_dir = f"tmp_{os.path.basename(audio_path).split('.')[0]}_wx_chunk_2m"
config_path = sys.argv[2] if len(sys.argv) > 2 else "configs/asr_final_from_chatui.yaml"
print(f"Transcribing audio: {audio_path} with config: {config_path}")
os.makedirs(output_dir, exist_ok=True)

# Load VAD model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vad_model = load_vad_model(device)

# VAD parameters
vad_params = {"vad_onset": 0.500, "vad_offset": 0.363}
chunk_size = 150  # Maximum chunk size in seconds
overlap = 30  # Overlap between chunks in seconds


def process_and_transcribe(audio_path: str, output_dir: str):
    """
    Processes the input audio with VAD and transcribes it using OpenAI's API.
    """
    audio = load_audio(audio_path)
    
    vad_segments = vad_model({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE})
    vad_segments = merge_chunks(vad_segments, chunk_size, onset=vad_params["vad_onset"], offset=vad_params["vad_offset"])

    # Transcribe each chunk
    merged_transcript = []
    segments: List[SingleSegment] = []
    
    for idx, segment in enumerate(vad_segments):
        start_ms = int(segment["start"] * 1000)
        end_ms = int(segment["end"] * 1000)
        audio_chunk = audio[int(start_ms * SAMPLE_RATE / 1000):int(end_ms * SAMPLE_RATE / 1000)]
        
        # Scale and prepare audio chunk
        audio_chunk = (audio_chunk * np.iinfo(np.int16).max).astype(np.int16)
        # Ensure even length for proper sample width alignment
        if len(audio_chunk) % 2 != 0:
            audio_chunk = audio_chunk[:-1]

        # Save chunk to a temporary file
        chunk_path = os.path.join(output_dir, f"chunk_{idx:02d}.mp3")
        AudioSegment(
            audio_chunk.tobytes(), 
            sample_width=2,
            frame_rate=SAMPLE_RATE, 
            channels=1
        ).export(chunk_path, format="mp3")
        
        # Transcribe chunk
        print(f"Transcribing chunk {idx + 1}/{len(vad_segments)}: {start_ms/1000}-{end_ms/1000} seconds")
        transcript = transcribe_audio_with_openai(chunk_path, config_path).strip("...").strip()
        merged_transcript.append(transcript)
        segments.append(
                {
                    "text": transcript,
                    "start": round(vad_segments[idx]['start'], 3),
                    "end": round(vad_segments[idx]['end'], 3)
                }
            )


        # Save transcript for the chunk
        chunk_transcript_path = os.path.join(output_dir, f"chunk_{idx:02d}_transcript.txt")
        with open(chunk_transcript_path, "w") as f:
            f.write(transcript)
    merged_transcript_path = os.path.join(output_dir, "merged_transcript.txt")
    with open(merged_transcript_path, "w") as f:
        f.write("\n----------\n".join(merged_transcript))
    print(f"Merged transcript saved at: {merged_transcript_path}")
    write(os.path.join(output_dir, "result.json"), segments)
    os.symlink(os.path.abspath(audio_path), os.path.join(output_dir, "original.wav"))

# Run the transcription process
process_and_transcribe(audio_path, output_dir)
