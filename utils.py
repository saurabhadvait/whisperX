import json
import os
import pickle
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from langdetect import detect
from pydub import AudioSegment


def get_language_from_transcript(transcript: str) -> str:
    if not transcript.strip():
        raise ValueError("Transcript is empty.")
    return detect(transcript)
    
@lru_cache(maxsize=None)
def get_preferred_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    elif torch.backends.mps.is_available(): return torch.device("mps")
    else: return torch.device("cpu")

def get_audio_duration(audio_file: str) -> float:
    audio = AudioSegment.from_file(audio_file)
    return len(audio) / 1000.0  # Duration in seconds

def write(file_path: str, data: Any, verbose: bool = True):
    file_path = Path(file_path)
    if file_path.suffix == ".json":
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    elif file_path.suffix == ".pkl":
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
    elif file_path.suffix == ".txt":
        with open(file_path, "w", encoding="utf-8") as f:
            print(data, file=f)
    else:
        raise ValueError(f"File type not supported: {file_path}")
    print(f"Wrote to {file_path}") if verbose else None


def read(file_path: str, verbose: bool = True) -> Any:
    file_path = Path(file_path)
    if file_path.suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif file_path.suffix == ".pkl":
        with open(file_path, "rb") as f:
            data = pickle.load(f)
    elif file_path.suffix == ".npy":
        data = np.load(file_path)
    elif file_path.suffix == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            data = f.read()
    else:
        raise ValueError(f"File type not supported: {file_path}")
    print(f"Read {file_path}") if verbose else None
    return data

def plot_scores(alignment_result: Dict[str, Any], args: Any):
    from matplotlib import pyplot as plt
    scores, times = zip(*[(w["score"], w["start"]) for w in alignment_result["segments"][0]["words"] if "score" in w])
    plt.plot(times, scores)
    plt.xlabel("Time (s)")
    plt.ylabel("Score")
    plt.savefig(os.path.join(args.data_dir, f"plot_{args.out_name}.png"))
    plt.show()


def plot_alignment_with_gradient(out: List[Dict[str, Any]], save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import rcParams
    rcParams['font.family'] = 'Arial Unicode MS'

    out = [item for item in out if all(key in item for key in ['word', 'start', 'end'])]

    words = [item['word'] for item in out]
    starts = [item['start'] for item in out]
    ends = [item['end'] for item in out]

    max_time = max(ends)
    heatmap = np.linspace(0, 1, int(max_time * 100)).reshape(1, -1)  # 1D gradient
    heatmap = np.tile(heatmap, (len(words), 1))  # Repeat gradient for all words

    fig, ax = plt.subplots(figsize=(20, 10))

    ax.imshow(heatmap, aspect='auto', origin='lower', cmap='viridis', extent=[0, max_time, 0, len(words)])

    for i, (word, start, end) in enumerate(zip(words, starts, ends)):
        ax.plot([start, end], [i + 0.5, i + 0.5], 'w-', linewidth=2)  # White alignment line
        ax.text((start + end) / 2, i + 0.5, word, color='white', ha='center', va='center', fontsize=10)

    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=6)
    ax.set_xlabel('Time(s)', fontsize=14)
    ax.set_title('Alignment Visualization', fontsize=16)
    ax.grid(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    
def download_video(url: str, output_dir: str, start_time: str = None, end_time: str = None) -> bool:
    time_range = ""
    if start_time or end_time:
        start_time = start_time if start_time else "0"
        time_range = f" --download-sections *{start_time}-{end_time if end_time else ''}"
    
    os.system(f"yt-dlp --output '{output_dir}/%(title)s.%(ext)s'{time_range} {url}")
    return True

def download_audio(url: str, output_dir: str, start_time: str = None, end_time: str = None) -> bool:    # time=01:02:03
    time_range = ""
    if start_time or end_time:
        start_time = start_time if start_time else "0"
        time_range = f" --download-sections *{start_time}-{end_time if end_time else ''}"
    command = f"yt-dlp -x --audio-format mp3 --audio-quality 0 --output '{output_dir}/%(title)s.%(ext)s'{time_range} {url}"
    print("Downloading audio with command:", command)
    os.system(command)
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download audio from URL")
    parser.add_argument("url", type=str, help="URL to download from")
    parser.add_argument("--output_dir", "-o", type=str, default=".", help="Output directory")
    parser.add_argument("--start_time", "-s", type=str, default=None, help="Start time")
    parser.add_argument("--end_time", "-e", type=str, default=None, help="End time")
    args = parser.parse_args()
    download_audio(args.url, args.output_dir, args.start_time, args.end_time)
    

if __name__ == "__main__":
    main()