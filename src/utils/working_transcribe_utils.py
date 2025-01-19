import base64
import os
from functools import partial
from typing import List, Tuple

from omegaconf import OmegaConf
from openai import OpenAI
from pydub import AudioSegment

from src.merge_utils import merge
from src.utils import bcolors, get_preferred_device, read, write

read = partial(read, verbose=False)
DEVICE = get_preferred_device()
TSR = 16000
CUSTOM_CHUNK_SIZES = [
    10*60*1000,  # 10 minutes
    5*60*1000,   # 5 minutes
    2*60*1000,   # 2 minutes
    # 60*1000      # 1 minute
]

###############################################################################
# 1) A single function to split a region of audio with a given chunk duration
###############################################################################
def split_audio_region(
    audio_path: str,
    tmp_dir: str,
    chunk_duration_ms: int,
    chunk_overlap_ms: int,
    start_ms: int = 0,
    end_ms: int = None,
    prefix: str = "chunk",
    remove_old: bool = True
) -> list[dict]:
    audio = AudioSegment.from_file(audio_path).set_frame_rate(TSR)
    if end_ms is None or end_ms > len(audio):
        end_ms = len(audio)
    region = audio[start_ms:end_ms]

    os.makedirs(tmp_dir, exist_ok=True)
    if remove_old:
        for f in os.listdir(tmp_dir):
            if f.startswith(prefix) and f.endswith(".mp3"):
                os.remove(os.path.join(tmp_dir, f))

    chunk_info = []
    step_size = chunk_duration_ms - chunk_overlap_ms
    pos, i = 0, 0
    while pos < len(region):
        c = region[pos : pos + chunk_duration_ms]
        out_path = os.path.join(tmp_dir, f"{prefix}_{i:02d}.mp3")
        c.export(out_path, format="mp3", codec="libmp3lame", parameters=["-ar", str(TSR), "-ac", "1"])
        chunk_info.append({
            "index": i,
            "start_ms": start_ms + pos,
            "end_ms": start_ms + pos + len(c),
            "path": out_path
        })
        i += 1
        pos += step_size
    print(f"{bcolors.OKCYAN}Split '{prefix}': {len(chunk_info)} chunks, chunk={chunk_duration_ms//60000} min{bcolors.ENDC}")
    return chunk_info

###############################################################################
# 2) A helper to transcribe and merge a given set of chunks
#    Returns (final_transcript, success_bool)
#    If we ever see overlap < min_match_length, we return (None, False).
###############################################################################
def try_transcribe_merge(
    chunk_info: list[dict],
    config_path: str,
    min_match_length: int = 10
) -> Tuple[str, bool]:
    final_txt = ""
    for i, ck in enumerate(chunk_info):
        txt_path = ck["path"].replace(".mp3", "_transcript.txt")
        if not os.path.exists(txt_path):
            t = transcribe_audio(ck["path"], config_path).strip("...")
            assert t, f"Transcription failed for {ck['path']}"
            write(txt_path, t)
        else:
            t = read(txt_path).strip()

        if not final_txt:
            final_txt = t
            continue

        # Attempt merge
        snippet_old = final_txt[-1000:]
        snippet_new = t[:1000]
        out = merge(snippet_old, snippet_new)
        if out["match_length"] < min_match_length:
            print(
                f"{bcolors.WARNING}Overlap {out['match_length']} < {min_match_length}. "
                "Triggering fallback.{bcolors.ENDC}"
            )
            return (None, False)  # overlap too small => fail
        final_txt = final_txt[:-1000] + out["merged"] + t[1000:]
    return (final_txt, True)

###############################################################################
# 3) Transcribe a single chunk of audio with your config
###############################################################################
def transcribe_audio(audio_path: str, config_path: str) -> str:
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    with open(audio_path, "rb") as f:
        audio_data = f.read()
    b64_data = base64.b64encode(audio_data).decode("utf-8")
    # Insert b64 data as your pipeline expects
    cfg["messages"][-1]["content"][-1]["input_audio"]["data"] = b64_data

    client = OpenAI()
    resp = client.chat.completions.create(
        model=cfg["model"],
        messages=cfg["messages"],
        modalities=cfg["modalities"],
        temperature=cfg["temperature"],
        max_completion_tokens=cfg["max_completion_tokens"],
        top_p=cfg["top_p"],
        frequency_penalty=cfg["frequency_penalty"],
        presence_penalty=cfg["presence_penalty"]
    ).to_dict()
    return resp["choices"][0]["message"]["content"]

###############################################################################
# 4) The main recursive function that tries each chunk size in turn
###############################################################################
def transcribe_segment(
    audio_path: str,
    config_path: str,
    tmp_dir: str,
    chunk_durations_ms: list[int] = CUSTOM_CHUNK_SIZES,
    overlap_ms: int = 30_000,
    start_ms: int = 0,
    end_ms: int = None,
    min_match_length: int = 10
) -> str:
    """
    Recursively transcribe [start_ms, end_ms] of 'audio_path' using chunk 
    sizes in 'chunk_durations_ms'. If merging fails for the first chunk size, 
    fallback to the next, etc.
    """
    if not chunk_durations_ms:
        raise RuntimeError("All chunk durations exhausted; cannot transcribe this region.")

    # 1) Use the first chunk size in the list
    chunk_size = chunk_durations_ms[0]
    # 2) Split the region
    if not os.path.isdir(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
    if not os.path.islink(os.path.join(tmp_dir, "original.wav")):
        os.symlink(os.path.abspath(audio_path), os.path.join(tmp_dir, "original.wav"))

    chunk_info = split_audio_region(
        audio_path=audio_path,
        tmp_dir=tmp_dir,
        chunk_duration_ms=chunk_size,
        chunk_overlap_ms=overlap_ms,
        start_ms=start_ms,
        end_ms=end_ms,
        prefix=f"try_{int(chunk_size/60000)}m",
        remove_old=True
    )

    # 3) Try to transcribe+merge
    result, success = try_transcribe_merge(chunk_info, config_path, min_match_length)
    if success:
        # On success, write final and return
        final_txt_path = os.path.join(tmp_dir, f"final_{chunk_size//60000}m.txt")
        write(final_txt_path, result)
        print(f"{bcolors.OKGREEN}Success with chunk size {chunk_size//60000}m => {final_txt_path}{bcolors.ENDC}")
        return result
    else:
        # 4) Fallback => remove these partial transcripts if you want
        print(f"{bcolors.WARNING}Fallback: trying next chunk size...{bcolors.ENDC}")
        return transcribe_segment(
            audio_path,
            config_path,
            tmp_dir,
            chunk_durations_ms[1:], # drop the first chunk size
            overlap_ms,
            start_ms,
            end_ms,
            min_match_length
        )

def main():
    transcribe_segment(
        audio_path="my_audio.mp3",
        config_path="my_asr_config.yaml",
        tmp_dir="results/my_audio",
        chunk_durations_ms=CUSTOM_CHUNK_SIZES,
        overlap_ms=30_000,
        start_ms=0,
        end_ms=None,
        min_match_length=10
    )

if __name__ == "__main__":
    main()