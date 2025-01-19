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

from src.transcribe_utils import (bcolors,
                                  transcribe_and_merge_with_partial_fallback)


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
    tmp_dir = f"results/{args.config.split('.')[0]}" + (f"_{args.out_dir_suffix}" if args.out_dir_suffix else "") + f"/{os.path.basename(args.audio_path).replace('.mp3', '')}" 
    print(f"{bcolors.OKGREEN}Output directory: {tmp_dir}{bcolors.ENDC}")
    if os.path.exists(tmp_dir):
        if args.fresh:
            print(f"{bcolors.WARNING}Removing existing directory {tmp_dir}{bcolors.ENDC}")
            os.system(f"rm -rf {tmp_dir}")
        
    # split_audio(args.audio_path, tmp_dir) if not os.path.islink(os.path.join(tmp_dir, "original.wav")) else None
    if args.transcribe:
        config_path = f"configs/{args.config}"
        os.makedirs(tmp_dir, exist_ok=True)
        os.system(f"cp {config_path} {tmp_dir}")
        print(f"{bcolors.OKBLUE}Using config: {config_path}{bcolors.ENDC}")
        transcribe_and_merge_with_partial_fallback(args.audio_path, config_path, tmp_dir)
        # transcribe_and_merge_chunks(tmp_dir, config_path)
        # transcribe_chunks(tmp_dir, config_path)
        # merge_transcripts(tmp_dir)