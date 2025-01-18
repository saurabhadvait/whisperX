import base64
import os
from functools import partial

from omegaconf import OmegaConf
from openai import OpenAI
from pydub import AudioSegment
from tqdm import tqdm

from .align import align_and_save
from .merge_utils import merge
from .utils import bcolors, detect_language, get_preferred_device, read, write

read = partial(read, verbose=False)

DEVICE = get_preferred_device()
TSR = 16000
INITIAL_CHUNK_DURATION_MS = 10 * 60 * 1000
FALLBACK_CHUNK_DURATION_MS = 3 * 60 * 1000
CHUNK_OVERLAP_MS = 30 * 1000

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
    """
    Split either the entire audio (if 'start_ms=0' and 'end_ms=None') 
    or just a subset region '[start_ms, end_ms]' using 'chunk_duration_ms' 
    with overlap 'chunk_overlap_ms'.

    - audio_path: path to the source .mp3 or .wav audio file
    - tmp_dir: output directory
    - chunk_duration_ms: how many milliseconds per chunk
    - chunk_overlap_ms: how many milliseconds to overlap between chunks
    - start_ms: region start in the original audio (default=0)
    - end_ms: region end in the original audio (default=None => end of audio)
    - prefix: how to name the output chunk files, e.g. "chunk" or "fix"
    - remove_old: whether to remove old .mp3 chunk files matching prefix

    Returns:
      A list of dicts:
        [
          {
            "index": <chunk_index>,
            "start_ms": <absolute ms in the full audio>,
            "end_ms": <absolute ms>,
            "path": "/path/to/chunk_XX.mp3"
          },
          ...
        ]
    """
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(TSR)

    # If end_ms not given => end of file
    if end_ms is None or end_ms > len(audio):
        end_ms = len(audio)

    # Extract only the sub-region if start/end are given
    region = audio[start_ms:end_ms]
    total_region_len = len(region)

    os.makedirs(tmp_dir, exist_ok=True)

    # Optionally remove old chunk files that share this prefix
    if remove_old:
        for f in os.listdir(tmp_dir):
            if f.startswith(prefix) and f.endswith(".mp3"):
                os.remove(os.path.join(tmp_dir, f))

    # We store chunk metadata in chunk_info:
    chunk_info = []
    i = 0

    # Step size: chunk_duration - overlap
    step_size = chunk_duration_ms - chunk_overlap_ms
    pos = 0

    while pos < total_region_len:
        # Slice out [pos, pos + chunk_duration]
        chunk = region[pos : pos + chunk_duration_ms]
        out_path = os.path.join(tmp_dir, f"{prefix}_{i:02d}.mp3")
        chunk.export(
            out_path,
            format="mp3",
            codec="libmp3lame",
            parameters=["-ar", TSR, "-ac", "1"]
        )
        abs_start = start_ms + pos
        abs_end = abs_start + len(chunk)

        chunk_info.append({
            "index": i,
            "start_ms": abs_start,
            "end_ms": abs_end,
            "path": out_path
        })
        i += 1
        pos += step_size
        if pos >= total_region_len:
            break

    print(
        f"Split audio {prefix} => {len(chunk_info)} chunks "
        f"({chunk_duration_ms/60000:.1f} min each, overlap {chunk_overlap_ms} ms)."
    )

    return chunk_info


def main_split(audio_path: str, tmp_dir: str):
    if not os.path.islink(os.path.join(tmp_dir, "original.wav")):
        os.symlink(os.path.abspath(audio_path), os.path.join(tmp_dir, "original.wav"))
    
    chunk_info = split_audio_region(
        audio_path=audio_path,
        tmp_dir=tmp_dir,
        chunk_duration_ms=INITIAL_CHUNK_DURATION_MS,
        chunk_overlap_ms=CHUNK_OVERLAP_MS,
        start_ms=0,
        end_ms=None,       # entire audio
        prefix="chunk",    # chunk_00.mp3 etc.
        remove_old=True
    )
    return chunk_info

def fallback_split_region(tmp_dir: str, region_start: int, region_end: int):
    """
    Re-split only the region [region_start, region_end] with smaller chunks.
    """
    fix_chunk_info = split_audio_region(
        audio_path=os.path.join(tmp_dir, "original.wav"),  # or your original mp3
        tmp_dir=tmp_dir,
        chunk_duration_ms=FALLBACK_CHUNK_DURATION_MS,
        chunk_overlap_ms=CHUNK_OVERLAP_MS,
        start_ms=region_start,
        end_ms=region_end,
        prefix="smaller_chunk",
        remove_old=True
    )
    return fix_chunk_info

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

def transcribe_and_merge_chunks(
    tmp_dir: str,
    config_path: str,
    chunk_info: list[dict],
    initial_chunk_duration: int,
    fallback_chunk_duration: int,
    min_match_length: int = 10,
) -> str:
    """
    Transcribe chunks one by one and merge them into 'final_transcript'.
    If a merge fails (overlap < min_match_length), only re-split the region 
    covering chunk i-1 and i with fallback_chunk_duration, 
    then re-transcribe & merge that smaller region.
    """
    final_transcript = ""

    i = 0
    while i < len(chunk_info):
        current_chunk = chunk_info[i]
        current_txt_path = current_chunk["path"].replace(".mp3", "_transcript.txt")

        # 1. Transcribe if needed
        if not os.path.exists(current_txt_path):
            transcript = transcribe_audio(current_chunk["path"], config_path).strip("...")
            assert transcript, f"Transcription failed for {current_chunk['path']}."
            write(current_txt_path, transcript)
        else:
            transcript = read(current_txt_path).strip()

        # 2. If this is the first chunk, just append
        if not final_transcript:
            final_transcript = transcript
            i += 1
            continue

        # 3. Otherwise, attempt to merge with the last 1000 chars from 'final_transcript'
        snippet_old = final_transcript[-1000:]
        snippet_new = transcript[:1000]
        out = merge(snippet_old, snippet_new)
        match_length = out["match_length"]

        if match_length < min_match_length:
            print(
                f"{bcolors.WARNING}Low overlap: {match_length} tokens between chunk {i-1} and {i}. "
                f"Re-splitting *only* that region with smaller chunk size...{bcolors.ENDC}"
            )

            # Identify start-end of the combined region [chunk i-1 start, chunk i end]
            prev_chunk = chunk_info[i-1]
            region_start = prev_chunk["start_ms"]
            region_end   = current_chunk["end_ms"]

            # 1) Re-split that region with fallback chunk duration
            sub_chunks = fallback_resplit_region(
                audio_path=os.path.join(tmp_dir, "original.wav"),
                tmp_dir=tmp_dir,
                start_ms=region_start,
                end_ms=region_end,
                fallback_chunk_duration=fallback_chunk_duration,
                prefix=f"fix_{i-1}_{i}"
            )

            # 2) Transcribe & do "mini merges" among these smaller sub-chunks
            mini_merged_text = ""
            for sub_ck in sub_chunks:
                sub_txt_path = sub_ck["path"].replace(".mp3", "_transcript.txt")
                if not os.path.exists(sub_txt_path):
                    sub_trans = transcribe_audio(sub_ck["path"], config_path).strip("...")
                    write(sub_txt_path, sub_trans)
                else:
                    sub_trans = read(sub_txt_path).strip()

                # Merge each new sub-chunk transcript into 'mini_merged_text'
                if not mini_merged_text:
                    mini_merged_text = sub_trans
                else:
                    snippet_old2 = mini_merged_text[-1000:]
                    snippet_new2 = sub_trans[:1000]
                    out2 = merge(snippet_old2, snippet_new2)
                    # If still small overlap, might handle recursively or just accept it
                    mini_merged_text = (
                        mini_merged_text[:-1000] 
                        + out2["merged"] 
                        + sub_trans[1000:]
                    )

            # 3) Now 'mini_merged_text' replaces the portion of final_transcript 
            #    that was chunk i-1. So let's remove the last chunk i-1 text 
            #    from final_transcript, then append the new sub-chunk text.
            #    Because we do not have the exact boundary in final_transcript, 
            #    an approximation is to remove e.g. the last 2000-3000 characters 
            #    or so. Or you might track chunk i-1 text length in a separate index.

            # For simplicity, let's remove e.g. the last 3000 characters
            # and hope that covers chunk i-1. In a robust system you'd track offsets.
            final_transcript = final_transcript[:-3000] if len(final_transcript) > 3000 else ""

            # Append our newly corrected region text
            final_transcript += mini_merged_text

            # 4) Because we've replaced chunk i-1 and chunk i with new sub-chunks,
            #    we can skip chunk i in the main loop to avoid duplication.
            i += 1
            i += 1  # skip the old chunk i
            continue

        # If overlap is fine, accept the merge in final_transcript
        merged_text = out["merged"]
        final_transcript = final_transcript[:-1000] + merged_text + transcript[1000:]
        i += 1

    # Write out final transcript
    full_path = os.path.join(tmp_dir, "full_transcript.txt")
    write(full_path, final_transcript)
    print(f"{bcolors.OKGREEN}Full transcript saved to {full_path}{bcolors.ENDC}")
    return final_transcript

def fallback_resplit_region(
    audio_path: str,  # The original .wav symlink
    tmp_dir: str,
    start_ms: int,
    end_ms: int,
    fallback_chunk_duration: int,
    prefix: str = "fix"
) -> list[dict]:
    """
    Re-split only the region [start_ms, end_ms] into smaller chunks 
    (fallback_chunk_duration) and return that chunk_info.
    """
    audio = AudioSegment.from_file(audio_path)  # The "original.wav" symlink
    region = audio[start_ms:end_ms]

    # Optionally remove old sub-chunks from that region if they exist:
    for f in os.listdir(tmp_dir):
        if f.startswith(f"{prefix}_") and f.endswith(".mp3"):
            os.remove(os.path.join(tmp_dir, f))

    sub_chunk_info = []
    i = 0
    # For smaller fallback splitting, might also choose an overlap if desired
    fallback_overlap = 0  # e.g. 0 or 10 seconds if you prefer
    step = fallback_chunk_duration - fallback_overlap

    for pos in range(0, len(region), step):
        sub_chunk = region[pos : pos + fallback_chunk_duration]
        sub_out_path = os.path.join(tmp_dir, f"{prefix}_sub_{i:02d}.mp3")
        sub_chunk.export(sub_out_path, format="mp3", codec="libmp3lame", parameters=["-ar", str(TSR), "-ac", "1"])

        sub_chunk_info.append({
            "index": i,
            "start_ms": start_ms + pos,       # relative to original audio
            "end_ms": start_ms + pos + len(sub_chunk),
            "path": sub_out_path,
        })
        i += 1
        if pos + fallback_chunk_duration >= len(region):
            break

    return sub_chunk_info


# def prev_align(tmp_dir: str):
#     print("Aligning...")
#     lang = detect_language(f"{tmp_dir}/chunk_00_transcript.txt")
#     for transcript_file in tqdm(sorted([f for f in os.listdir(tmp_dir) if f.endswith("_transcript.txt")]), desc="Aligning"):
#         transcript_path = os.path.join(tmp_dir, transcript_file)
#         out_file = transcript_path.replace("_transcript.txt", "_timestamps.json")
        
#         if not os.path.exists(out_file):
#             align_and_save(
#                 audio_file=transcript_path.replace("_transcript.txt", ".mp3"),
#                 transcript_file=transcript_path,
#                 out_file=out_file,
#                 device=DEVICE,
#                 language_code=lang,
#             )
#             print(f"Timestamps saved to {out_file}.")
#         else:
#             print(f"Timestamps already exist at {out_file}.")


# def prev_split_audio(audio_path: PATH_LIKE, tmp_dir: PATH_LIKE):
#     audio = AudioSegment.from_mp3(audio_path)
#     audio = audio.set_frame_rate(TSR)
    
#     os.makedirs(tmp_dir, exist_ok=True)
    
#     for i, start in enumerate(range(0, len(audio), CHUNK_DURATION - CHUNK_OVERLAP)):
#         chunk = audio[start:start + CHUNK_DURATION]
#         out_path = os.path.join(tmp_dir, f"chunk_{i:02d}.mp3")
#         chunk.export(out_path, format="mp3", codec="libmp3lame", parameters=["-ar", str(TSR), "-ac", "1"])
#         if start + CHUNK_DURATION >= len(audio):
#             break
            
#     print(f"Audio split into {len(os.listdir(tmp_dir))} chunks at {TSR}Hz.")
#     os.symlink(os.path.abspath(audio_path), os.path.join(tmp_dir, "original.wav"))


# def split_audio(audio_path: PATH_LIKE, tmp_dir: PATH_LIKE, chunk_duration: int) -> list[dict]:
#     audio = AudioSegment.from_mp3(audio_path)
#     audio = audio.set_frame_rate(TSR)
    
#     os.makedirs(tmp_dir, exist_ok=True)
    
#     # Clean out old mp3 chunks from previous runs
#     for f in os.listdir(tmp_dir):
#         if f.startswith("chunk_") and f.endswith(".mp3"):
#             os.remove(os.path.join(tmp_dir, f))

#     chunk_info = []
#     i = 0
#     for start in range(0, len(audio), chunk_duration - CHUNK_OVERLAP):
#         # slice the audio
#         chunk = audio[start : start + chunk_duration]
#         out_path = os.path.join(tmp_dir, f"chunk_{i:02d}.mp3")
#         chunk.export(out_path, format="mp3", codec="libmp3lame", parameters=["-ar", str(TSR), "-ac", "1"])
        
#         chunk_info.append({
#             "index": i,
#             "start_ms": start,
#             "end_ms": start + len(chunk),
#             "path": out_path,
#         })
        
#         i += 1
#         if start + chunk_duration >= len(audio):
#             break
    
#     # Make a symlink for the full audio if it doesn't exist
#     if not os.path.islink(os.path.join(tmp_dir, "original.wav")):
#         os.symlink(os.path.abspath(audio_path), os.path.join(tmp_dir, "original.wav"))

#     print(f"Audio split into {len(chunk_info)} chunks at {TSR}Hz.")
#     return chunk_info
