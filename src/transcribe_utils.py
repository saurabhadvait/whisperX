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

INITIAL_CHUNK_DURATION_MS = 10 * 60 * 1000
CHUNK_OVERLAP_MS = 30 * 1000

FALLBACK_SIZES = [
    5 * 60 * 1000,  # 5min
    2 * 60 * 1000,  # 2min
    60 * 1000,      # 1min
]

def split_audio_region(
    audio_path: str,
    tmp_dir: str,
    chunk_duration_ms: int,
    chunk_overlap_ms: int,
    start_ms: int = 0,
    end_ms: int = None,
    prefix: str = "chunk",
) -> List[dict]:
    audio = AudioSegment.from_file(audio_path).set_frame_rate(TSR)
    if end_ms is None or end_ms > len(audio):
        end_ms = len(audio)

    region = audio[start_ms:end_ms]
    os.makedirs(tmp_dir, exist_ok=True)

    step_size = chunk_duration_ms - chunk_overlap_ms
    chunk_info = []
    pos = 0
    i = 0
    while pos < len(region):
        c = region[pos : pos + chunk_duration_ms]
        out_path = os.path.join(tmp_dir, f"{prefix}_{i:02d}.mp3")
        c.export(
            out_path, format="mp3", codec="libmp3lame", 
            parameters=["-ar", str(TSR), "-ac", "1"]
        )
        chunk_info.append({
            "index": i,
            "start_ms": start_ms + pos,
            "end_ms": start_ms + pos + len(c),
            "path": out_path
        })
        i += 1
        pos += step_size

    print(
        f"{bcolors.OKCYAN}Split '{prefix}': {len(chunk_info)} chunks of {chunk_duration_ms//60000}m, overlap={chunk_overlap_ms//1000}s{bcolors.ENDC}"
    )
    return chunk_info

def fallback_subregion(
    audio_path: str,
    config_path: str,
    tmp_dir: str,
    start_ms: int,
    end_ms: int,
    fallback_sizes: List[int],
    min_match_length: int,
    prefix: str = "fix"
) -> str:
    if not fallback_sizes:
        raise RuntimeError("No fallback sizes left for partial region fallback.")

    size = fallback_sizes[0]
    sub_chunks = split_audio_region(
        audio_path,
        tmp_dir,
        chunk_duration_ms=size,
        chunk_overlap_ms=CHUNK_OVERLAP_MS,
        start_ms=start_ms,
        end_ms=end_ms,
        prefix=prefix,
    )

    sub_result, success = try_transcribe_merge(sub_chunks, config_path, min_match_length)
    if success:
        return sub_result
    else:
        print(
            f"{bcolors.WARNING}[fallback_subregion] Overlap fail with chunk size="
            f"{size/60000:0.1f}m, trying next smaller...{bcolors.ENDC}"
        )
        return fallback_subregion(
            audio_path,
            config_path,
            tmp_dir,
            start_ms,
            end_ms,
            fallback_sizes[1:],
            min_match_length,
            prefix
        )

def try_transcribe_merge(
    chunk_info: List[dict],
    config_path: str,
    min_match_length: int = 10
) -> Tuple[str, bool]:
    merged_txt = ""
    for ck in chunk_info:
        txt_path = ck["path"].replace(".mp3", "_transcript.txt")
        if not os.path.exists(txt_path):
            print(f"{bcolors.OKBLUE}Transcribing from {ck['start_ms']/60000:0.1f}m to {ck['end_ms']/60000:0.1f}m{bcolors.ENDC}")
            t = transcribe_audio(ck["path"], config_path).strip("...")
            if not t:
                return (None, False)
            write(txt_path, t)
        else:
            print(f"{bcolors.WARNING}Transcript already exists for {ck['path']} => skipping{bcolors.ENDC}")
            t = read(txt_path).strip()

        if not merged_txt:
            merged_txt = t
            continue

        snippet_old = merged_txt[-1000:]
        snippet_new = t[:1000]
        out = merge(snippet_old, snippet_new)
        ml = out["match_length"]
        if ml < min_match_length:
            return (None, False)  # fail
        merged_txt = merged_txt[:-1000] + out["merged"] + t[1000:]
    return (merged_txt, True)

def transcribe_audio(audio_path: str, config_path: str) -> str:
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    with open(audio_path, "rb") as f:
        data = f.read()
    aud_b64 = base64.b64encode(data).decode("utf-8")
    # Insert base64 data into config
    cfg["messages"][-1]["content"][-1]["input_audio"]["data"] = aud_b64

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

def transcribe_and_merge_with_partial_fallback(
    audio_path: str,
    config_path: str,
    tmp_dir: str,
    initial_chunk_size: int = INITIAL_CHUNK_DURATION_MS,
    overlap_ms: int = CHUNK_OVERLAP_MS,
    fallback_sizes: List[int] = FALLBACK_SIZES,
    min_match_length: int = 10
) -> str:
    os.makedirs(tmp_dir, exist_ok=True)
    if not os.path.islink(os.path.join(tmp_dir, "original.wav")):
        audio = AudioSegment.from_file(audio_path)
        print(f"{bcolors.BOLD}Audio duration: {len(audio)/60000:0.1f} min{bcolors.ENDC}")
        os.symlink(os.path.abspath(audio_path), os.path.join(tmp_dir, "original.wav"))

    chunk_info = split_audio_region(
        audio_path=audio_path,
        tmp_dir=tmp_dir,
        chunk_duration_ms=initial_chunk_size,
        chunk_overlap_ms=overlap_ms,
        prefix="main",
    )

    # We'll keep two buffers:
    # safe_merged = fully locked-in transcript
    # unsafe_merged = the tail end that might still be replaced if a fallback is triggered
    safe_merged = ""
    unsafe_merged = ""

    i = 0
    while i < len(chunk_info):
        ck = chunk_info[i]
        txt_path = ck["path"].replace(".mp3", "_transcript.txt")

        if not os.path.exists(txt_path):
            print(f"{bcolors.OKBLUE}Transcribing {initial_chunk_size//60000}m from {ck['start_ms']/60000:0.1f}m to {ck['end_ms']/60000:0.1f}m{bcolors.ENDC}")
            t = transcribe_audio(ck["path"], config_path).strip().strip("...")
            if not t:
                raise RuntimeError(f"Empty transcript for {ck['path']}")
            write(txt_path, t)
        else:
            t = read(txt_path).strip()

        # If unsafe_merged is empty => this is the first chunk
        if not unsafe_merged:
            unsafe_merged = t
            i += 1
            continue

        out = merge(unsafe_merged[-1000:], t[:1000])
        ml = out["match_length"]
        if ml < min_match_length:
            # Merge fail => partial fallback on [chunk i-1, chunk i]
            print(f"{bcolors.WARNING}Low overlap of only {ml} tokens => fallback for chunk {i-1} & {i}{bcolors.ENDC}")
            region_start = chunk_info[i-1]["start_ms"]
            region_end   = ck["end_ms"]

            # Re-split & re-transcribe just that region with fallback sizes
            sub_result = fallback_subregion(
                audio_path=os.path.join(tmp_dir, "original.wav"),
                config_path=config_path,
                tmp_dir=tmp_dir,
                start_ms=region_start,
                end_ms=region_end,
                fallback_sizes=fallback_sizes,
                min_match_length=min_match_length,
                prefix=f"fix_{region_start/60000:0.1f}m_{region_end/60000:0.1f}m"
            )

            # Replace entire unsafe_merged with sub_result,
            # because chunk i-1 wasn't "safe" yet
            unsafe_merged = sub_result
            # skip chunk i
            i += 1
            continue
        else:
            # Good overlap => incorporate into unsafe_merged
            merged_text = (
                unsafe_merged[:-1000] + out["merged"] + t[1000:]
                if len(unsafe_merged) > 1000 else out["merged"] + t[1000:]
            )
            unsafe_merged = merged_text

        # Move the safe portion from unsafe_merged => safe_merged
        # keep only last 1000 in unsafe_merged
        if len(unsafe_merged) > 1000:
            safe_merged += unsafe_merged[:-1000]
            unsafe_merged = unsafe_merged[-1000:]

        i += 1

    # Combine final transcript
    final_transcript = safe_merged + unsafe_merged
    out_path = os.path.join(tmp_dir, "full_transcript.txt")
    write(out_path, final_transcript)
    print(f"{bcolors.OKGREEN}Final transcript saved => {out_path}{bcolors.ENDC}")
    return final_transcript

def main():
    transcribe_and_merge_with_partial_fallback(
        audio_path="my_audio.mp3",
        config_path="my_asr_config.yaml",
        tmp_dir="results/my_audio",
        initial_chunk_size=INITIAL_CHUNK_DURATION_MS,
        overlap_ms=CHUNK_OVERLAP_MS,
        fallback_sizes=FALLBACK_SIZES,
        min_match_length=10
    )

if __name__ == "__main__":
    main()
