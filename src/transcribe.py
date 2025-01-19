import os
from argparse import ArgumentParser

from tqdm import tqdm

from src.align import add_speaker_diarization, align_and_save
from src.transcribe_utils import (bcolors, detect_language,
                                  transcribe_and_merge_with_partial_fallback)
from src.utils import (bcolors, detect_language, get_preferred_device, read,
                       write)

DEVICE = get_preferred_device()
def align(tmp_dir: str):
    transcript_file = os.path.join(tmp_dir, "complete_transcript.txt")
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

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the audio file.")
    parser.add_argument("--config", type=str, default="asr_final_from_chatui.yaml", help="Name of the configuration file.") # best is asr_final_from_chatui.yaml
    parser.add_argument("--suffix", type=str, default="", help="Suffix to append to the output directory.")
    parser.add_argument("--num_speakers", type=int, default=None, help="Number of speakers to diarize.")
    parser.add_argument("-t", "--transcribe", action="store_true", help="Run transcription.")
    parser.add_argument("-a", "--align", action="store_true", help="Run alignment after transcription.")
    parser.add_argument("-d", "--diarize", action="store_true", help="Run speaker diarization.")
    parser.add_argument("-f", "--fresh", action="store_true", help="Run everything from scratch.")
    args = parser.parse_args()
    assert args.audio_path.endswith(".mp3"), "Only MP3 files are supported."
    return args

if __name__ == "__main__":
    args = parse_args()     
    tmp_dir = f"results/{args.config.split('.')[0]}" + (f"_{args.suffix}" if args.suffix else "") + f"/{os.path.basename(args.audio_path).replace('.mp3', '')}" 
    print(f"{bcolors.OKGREEN}Output directory: {tmp_dir}{bcolors.ENDC}")
    if os.path.exists(tmp_dir):
        if args.fresh:
            print(f"{bcolors.WARNING}Removing existing directory {tmp_dir}{bcolors.ENDC}")
            os.system(f"rm -rf {tmp_dir}")
        
    if args.transcribe:
        config_path = f"configs/{args.config}"
        os.makedirs(tmp_dir, exist_ok=True)
        os.system(f"cp {config_path} {tmp_dir}")
        print(f"{bcolors.OKBLUE}Using config: {config_path}{bcolors.ENDC}")
        transcribe_and_merge_with_partial_fallback(args.audio_path, config_path, tmp_dir)
        
    if args.align:
        align(tmp_dir)
    
    if args.diarize:
        diarize(tmp_dir, args.num_speakers)
