import base64
import json
import os
import time
from argparse import ArgumentParser
from functools import partial
from typing import Optional

from omegaconf import OmegaConf
from openai import OpenAI
from pydub import AudioSegment
from tqdm import tqdm

from align import add_speaker_diarization, align_and_save
from merge_utils import merge
from utils import (PATH_LIKE, bcolors, detect_language, get_preferred_device,
                   read, write)

read = partial(read, verbose=False)

TSR = 8000
CHUNK_DURATION = int(5 * 60 * 1000)  # in milliseconds
CHUNK_OVERLAP = int(0.5 * 60 * 1000)
DEVICE = get_preferred_device()

def transcribe_audio(audio_path: str, config_path: str) -> str:
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
    audio_data_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    config["messages"][-1]["content"][-1]["input_audio"]["data"] = audio_data_base64

    client = OpenAI()
    response = client.chat.completions.create(
        model=config["model"],
        messages=config["messages"],
        modalities=config["modalities"],
        temperature=config["temperature"],
        max_completion_tokens=config["max_completion_tokens"],
        top_p=config["top_p"],
        frequency_penalty=config["frequency_penalty"],
        presence_penalty=config["presence_penalty"]
    )
    return response.to_dict()["choices"][0]["message"]["content"]


def split_audio(audio_path: PATH_LIKE, tmp_dir: PATH_LIKE):
    """Split audio into chunks with specified sample rate"""
    audio = AudioSegment.from_mp3(audio_path)
    audio = audio.set_frame_rate(TSR)
    
    os.makedirs(tmp_dir, exist_ok=True)
    
    for i, start in enumerate(range(0, len(audio), CHUNK_DURATION - CHUNK_OVERLAP)):
        chunk = audio[start:start + CHUNK_DURATION]
        out_path = os.path.join(tmp_dir, f"chunk_{i:02d}.mp3")
        chunk.export(out_path, format="mp3", codec="libmp3lame", parameters=["-ar", str(TSR), "-ac", "1"])
        if start + CHUNK_DURATION >= len(audio):
            break
            
    print(f"Audio split into {len(os.listdir(tmp_dir))} chunks at {TSR}Hz.")
    os.symlink(os.path.abspath(audio_path), os.path.join(tmp_dir, "original.wav"))

def transcribe_chunks(tmp_dir: PATH_LIKE, config_path: PATH_LIKE):
    for chunk_file in tqdm(sorted([f for f in os.listdir(tmp_dir) if f.endswith(".mp3")]), desc="Transcribing"):
        chunk_path = os.path.join(tmp_dir, chunk_file)
        start_time = time.time()
        transcript = transcribe_audio(chunk_path, config_path=config_path).strip("...")
        time_taken = time.time() - start_time

        with open(chunk_path.replace(".mp3", "_transcript.txt"), "w") as f:
            f.write(transcript)
        
        print(f"Transcript saved to {chunk_path.replace('.mp3', '_transcript.txt')} in {time_taken:.2f} seconds.")

def prev_align(tmp_dir: str):
    print("Aligning...")
    lang = detect_language(f"{tmp_dir}/chunk_00_transcript.txt")
    for transcript_file in tqdm(sorted([f for f in os.listdir(tmp_dir) if f.endswith("_transcript.txt")]), desc="Aligning"):
        transcript_path = os.path.join(tmp_dir, transcript_file)
        out_file = transcript_path.replace("_transcript.txt", "_timestamps.json")
        
        if not os.path.exists(out_file):
            align_and_save(
                audio_file=transcript_path.replace("_transcript.txt", ".mp3"),
                transcript_file=transcript_path,
                out_file=out_file,
                device=DEVICE,
                language_code=lang,
            )
            print(f"Timestamps saved to {out_file}.")
        else:
            print(f"Timestamps already exist at {out_file}.")

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

def merge_transcripts(tmp_dir: str) -> str:
    final_transcript = ""
    for f in sorted(f for f in os.listdir(tmp_dir) if f.startswith("chunk_") and f.endswith("_transcript.txt")):
        next_text = read(os.path.join(tmp_dir, f)).strip()
        if not final_transcript:
            final_transcript = next_text
            continue
        final_transcript = final_transcript[:-1000] + merge(final_transcript[-1000:], next_text[:1000])["merged"] + next_text[1000:]
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
    if not args.out_dir_suffix:
        args.out_dir_suffix = args.config.split(".")[0]   
    tmp_dir = f"new/best/{os.path.basename(args.audio_path).replace('.mp3', '')}{args.out_dir_suffix}"
    if os.path.exists(tmp_dir):
        if args.fresh:
            print(f"{bcolors.WARNING}Removing existing temporary directory {tmp_dir}.{bcolors.ENDC}")
            os.system(f"rm -rf {tmp_dir}")
        
    split_audio(args.audio_path, tmp_dir) if not os.path.islink(os.path.join(tmp_dir, "original.wav")) else None
    if args.transcribe:
        config_path = f"configs/{args.config}"
        print(f"{bcolors.OKBLUE}Using config: {config_path}{bcolors.ENDC}")
        transcribe_chunks(tmp_dir, config_path)
        merge_transcripts(tmp_dir)
    
    if args.align:
        align(tmp_dir)
    
    if args.diarize:
        diarize(tmp_dir, args.num_speakers)
