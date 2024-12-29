## audio duration >= transcript duration
import argparse
import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams['font.family'] = 'Arial Unicode MS'

from utils import read, write
from whisperx.new_alignment import align, load_align_model, load_audio


def to_json(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        text = result["segments"][0]["text"]
        text = text.replace("'", '"')
        out = json.loads(text)[0]["words"]
    except Exception as e:
        print("First type of processing failed, trying word_segments directly...")
        out = result["word_segments"]
    except Exception as e:
        print("Error in post-processing: ", e)
        return None
    out[0]["start"] = out[0]["end"] - .2     # hack: start is 200 ms before the end of first word
    return out

def load_and_align(audio_file, transcript_file, out_file, device="cpu", language_code="hi") -> List[Dict[str, Any]]:
    audio = load_audio(audio_file)
    audio_duration = audio.shape[-1] / 16000.0  # Duration in seconds
    model_a, metadata = load_align_model(language_code=language_code, device=device, model_dir="models/")

    transcript = read(transcript_file)

    alignment_result = align(
        [{"text": transcript, 'start': 0, 'end': audio_duration}],
        model_a, metadata, audio, device, return_char_alignments=False
    )
    out = to_json(alignment_result)
    if out:
        write(out_file, out)
        print(f"Alignment completed. Output saved to {out_file}.")
    else:
        breakpoint()
    return out

def plot_alignment_with_gradient(out: List[Dict[str, Any]], save_path=None):
    """
    Plot the word alignments with a gradient background and display Hindi text.
    
    Inputs:
        out: List of dictionaries containing word timing information.
             Each dict has 'word', 'start', and 'end' keys.
    """
    out = [item for item in out if all(key in item for key in ['word', 'start', 'end'])]

    # Extract word info
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
    ax.set_title('Viz', fontsize=16)
    ax.grid(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path) 
    plt.show()
        
def main():
    parser = argparse.ArgumentParser(description="WhisperX Audio Alignment Script")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory.")
    parser.add_argument("--out_name", type=str, default=None, help="Output file.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Device to run the model.")
    parser.add_argument("--language_code", type=str, default="hi", help="Language code for alignment (e.g., 'hi' for Hindi).")

    args = parser.parse_args()
    audio_file = None
    transcript_file = None
    for file in os.listdir(args.data_dir):
        if file.endswith(".mp3"):
            audio_file = os.path.join(args.data_dir, file)
        elif file.endswith(".txt"):
            transcript_file = os.path.join(args.data_dir, file)
    if not args.out_name:
        args.out_name = os.path.basename(args.data_dir)
    out_file = os.path.join(args.data_dir, f"{args.out_name}.json")

    assert audio_file is not None, f"No audio file found in {args.data_dir}."
    assert transcript_file is not None, f"No transcript file found in {args.data_dir}."
    out = load_and_align(
        audio_file=audio_file,
        transcript_file=transcript_file,
        out_file=out_file,
        device=args.device,
        language_code=args.language_code,
    )
    plot_alignment_with_gradient(out, save_path=os.path.join(args.data_dir, f"plot_{args.out_name}.png"))
    
if __name__ == "__main__":
    main()
