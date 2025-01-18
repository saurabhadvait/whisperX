import base64
import json
import os
import time
from argparse import ArgumentParser
from functools import partial
from typing import Optional

from openai import OpenAI
from pydub import AudioSegment
from tqdm import tqdm

from src.align import add_speaker_diarization, align_and_save
from src.merge_utils import merge
from src.utils import (PATH_LIKE, bcolors, detect_language,
                       get_preferred_device, read, write)
from src.working_transcribe_utils import DEVICE, transcribe_segment


def transcribe_chunks(tmp_dir: PATH_LIKE, config_path: PATH_LIKE):
    for chunk_file in tqdm(sorted([f for f in os.listdir(tmp_dir) if f.endswith(".mp3")]), desc="Transcribing"):
        chunk_path = os.path.join(tmp_dir, chunk_file)
        start_time = time.time()
        transcript = transcribe_audio(chunk_path, config_path=config_path).strip("...")
        time_taken = time.time() - start_time
        write(chunk_path.replace(".mp3", "_transcript.txt"), transcript)        
        print(f"Transcript saved to {chunk_path.replace('.mp3', '_transcript.txt')} in {time_taken:.2f} seconds.")

def align(tmp_dir: str):
    transcript_file = os.path.join(tmp_dir, "full_transcript.txt")
    align_and_save(os.path.join(tmp_dir, "original.wav"), transcript_file, os.path.join(tmp_dir, "full_timestamps.json"), device=DEVICE, language_code=detect_language(transcript_file))

def diarize(tmp_dir: str, num_speakers: int | None):
    for timestamp_file in tqdm(sorted([f for f in os.listdir(tmp_dir) if f.endswith("_timestamps.json")]), desc="Diarizing"):
        timestamp_path = os.path.join(tmp_dir, timestamp_file)
        out_file = timestamp_path.replace("_timestamps.json", "_diarized.json")
        add_speaker_diarization(
            audio=timestamp_path.replace("_timestamps.json", ".mp3"),
            out_file=out_file,
            alignment_result=read(timestamp_path),
            device=DEVICE,
            num_speakers=num_speakers,
        )
        print(f"Diarization saved to {out_file}.")

def concat_transcripts(tmp_dir: str) -> str:
    final_transcript = "\n\n----------------------------------------\n\n".join(
        open(os.path.join(tmp_dir, f)).read().strip()
        for f in sorted(f for f in os.listdir(tmp_dir) if f.endswith("_transcript.txt"))
    )
    write(os.path.join(tmp_dir, "full_transcript.txt"), final_transcript)
    return final_transcript

def prev_merge_transcripts(tmp_dir: str) -> str:
    final_transcript = ""
    for f in sorted(f for f in os.listdir(tmp_dir) if f.startswith("chunk_") and f.endswith("_transcript.txt")):
        next_text = read(os.path.join(tmp_dir, f)).strip()
        if not final_transcript:
            final_transcript = next_text
            continue
        out = merge(final_transcript[-1000:], next_text[:1000])
        final_transcript = final_transcript[:-1000] + out["merged"] + next_text[1000:]
        match_length = out["match_length"]
        if match_length < 10:
            print(f"WARNING: Low overlap found between the two transcripts: {match_length} tokens. Probably error in transcripts. Check them...")
    write(os.path.join(tmp_dir, "full_transcript.txt"), final_transcript)
    return final_transcript

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the audio file.")
    parser.add_argument("--config", type=str, default="asr_final_from_chatui.yaml", help="Name of the configuration file.") # best is asr_final_from_chatui.yaml
    parser.add_argument("--out_dir_suffix", type=str, default="", help="Suffix to append to the output directory.")
    parser.add_argument("--num_speakers", type=int, default=None, help="Number of speakers to diarize.")
    parser.add_argument("-t", "--transcribe", action="store_true", help="Run transcription.")
    parser.add_argument("-a", "--align", action="store_true", help="Run alignment after transcription.")
    parser.add_argument("-d", "--diarize", action="store_true", help="Run speaker diarization.")
    parser.add_argument("--fresh", action="store_true", help="Run everything from scratch.")
    args = parser.parse_args()
    assert args.audio_path.endswith(".mp3"), "Only MP3 files are supported."
    return args

if __name__ == "__main__":
    args = parse_args()     
    tmp_dir = f"results/{args.config.split('.')[0]}/{os.path.basename(args.audio_path).replace('.mp3', '')}" + (f"_{args.out_dir_suffix}" if args.out_dir_suffix else "")
    print(f"{bcolors.OKGREEN}Output directory: {tmp_dir}{bcolors.ENDC}")
    if os.path.exists(tmp_dir):
        if args.fresh:
            print(f"{bcolors.WARNING}Removing existing directory {tmp_dir}{bcolors.ENDC}")
            os.system(f"rm -rf {tmp_dir}")
        
    # split_audio(args.audio_path, tmp_dir) if not os.path.islink(os.path.join(tmp_dir, "original.wav")) else None
    if args.transcribe:
        config_path = f"configs/{args.config}"
        print(f"{bcolors.OKBLUE}Using config: {config_path}{bcolors.ENDC}")
        transcribe_segment(args.audio_path, config_path, tmp_dir)
        # transcribe_and_merge_chunks(tmp_dir, config_path)
        # transcribe_chunks(tmp_dir, config_path)
        # merge_transcripts(tmp_dir)
    
    if args.align:
        align(tmp_dir)
    
    if args.diarize:
        diarize(tmp_dir, args.num_speakers)
