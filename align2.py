import argparse
import gc
import os

import torch
import whisperx
from pydub import AudioSegment


def get_audio_duration(audio_file):
    """
    Get the duration of an audio file in seconds.
    """
    audio = AudioSegment.from_file(audio_file)
    return len(audio) / 1000.0  # Duration in seconds


def split_audio(audio_file, chunk_duration_ms=30000):
    """
    Split the audio file into smaller chunks.
    """
    audio = AudioSegment.from_file(audio_file)
    chunks = [audio[i:i + chunk_duration_ms] for i in range(0, len(audio), chunk_duration_ms)]
    return chunks


def save_chunks(chunks, output_dir="chunks", base_name="chunk"):
    """
    Save the audio chunks to files.
    """
    os.makedirs(output_dir, exist_ok=True)
    chunk_paths = []
    for i, chunk in enumerate(chunks):
        chunk_path = f"{output_dir}/{base_name}_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)
    return chunk_paths


def load_and_align(audio_chunk, transcript, model_a, metadata, device):
    """
    Align a single audio chunk.
    """
    audio = whisperx.load_audio(audio_chunk)
    duration = len(AudioSegment.from_file(audio_chunk)) / 1000.0  # Duration in seconds

    alignment_result = whisperx.align(
        [{"text": transcript, 'start': 0, 'end': duration}],
        model_a, metadata, audio, device, return_char_alignments=False
    )
    return alignment_result["segments"]


def process_audio_chunks(audio_file, transcript_file, out_file, device="cpu", language_code="hi", chunk_duration_ms=120000):
    """
    Process an audio file by splitting it into chunks and aligning each chunk.
    """
    # Split the audio file into chunks
    chunks = split_audio(audio_file, chunk_duration_ms=chunk_duration_ms)
    chunk_paths = save_chunks(chunks)

    # Load transcript
    with open(transcript_file, "r") as f:
        transcript = f.read()

    # Load alignment model
    model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device, model_dir="models/")

    # Process each chunk
    all_segments = []
    for chunk_path in chunk_paths:
        print(f"Processing chunk: {chunk_path}")
        segments = load_and_align(chunk_path, transcript, model_a, metadata, device)
        all_segments.extend(segments)

    # Save aligned segments to the output file
    with open(out_file, "w") as f:
        for segment in all_segments:
            f.write(f"{segment}\n")

    print(f"Alignment completed. Output saved to {out_file}.")

    # Cleanup
    # for chunk_path in chunk_paths:
    #     os.remove(chunk_path)  # Remove temporary chunk files
    

def main():
    parser = argparse.ArgumentParser(description="WhisperX Audio Alignment Script with Chunking")
    parser.add_argument("--audio_file", type=str, required=True, help="Path to the input audio file.")
    parser.add_argument("--transcript_file", type=str, required=True, help="Path to the transcript file.")
    parser.add_argument("--out_file", type=str, default=None, help="Path to save the alignment output.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Device to run the model.")
    parser.add_argument("--language_code", type=str, default="hi", help="Language code for alignment (e.g., 'hi' for Hindi).")
    parser.add_argument("--chunk_duration_ms", type=int, default=30000, help="Duration of each chunk in milliseconds.")

    args = parser.parse_args()
    if args.out_file is None:
        args.out_file = args.transcript_file.replace(".txt", "_aligned.txt")
        print(f"Output file not provided. Saving output to {args.out_file}.")

    process_audio_chunks(
        audio_file=args.audio_file,
        transcript_file=args.transcript_file,
        out_file=args.out_file,
        device=args.device,
        language_code=args.language_code,
        chunk_duration_ms=args.chunk_duration_ms,
    )


if __name__ == "__main__":
    main()
    main()
