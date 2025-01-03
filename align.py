import argparse
import json
import os
from typing import Any, Dict, List

import whisperx
from utils import plot_alignment_with_gradient, read, write
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
    out[0]["start"] = out[0]["end"] - 0.2  # Hack: start is 200 ms before the end of the first word
    return out


def load_and_align(audio, transcript_file, out_file, device="cpu", language_code="hi", write_complete_result=True) -> List[Dict[str, Any]]:
    audio_duration = audio.shape[-1] / 16000.0  # Duration in seconds
    model_a, metadata = load_align_model(language_code=language_code, device=device, model_dir="models/")

    transcript = read(transcript_file)

    alignment_result = align(
        [{"text": transcript, 'start': 0, 'end': audio_duration}],
        model_a, metadata, audio, device, return_char_alignments=False
    )
    if write_complete_result:
        out = alignment_result
    else:
        out = to_json(alignment_result)
    if out:
        write(out_file, out)
        print(f"Alignment completed. Output saved to {out_file}.")
    else:
        breakpoint()
    return out


def add_speaker_diarization(audio, alignment_result, device="cpu", num_speakers=None, min_speakers=None, max_speakers=None):
    print("Running speaker diarization...")
    diarize_model = whisperx.DiarizationPipeline(device=device)
    diarize_segments = diarize_model(audio, num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers)
    diarize_segments.to_json("diarize.json", indent=2)
    # Assign speaker labels to words in the alignment result
    result_with_speakers = whisperx.assign_word_speakers(diarize_segments, alignment_result)
    print("Speaker diarization completed.")
    return result_with_speakers


def main():
    parser = argparse.ArgumentParser(description="WhisperX Audio Alignment Script with Speaker Identification")
    parser.add_argument("--data", "-d", type=str, required=True, help="Path to the data directory or audio file.")
    parser.add_argument("--out_name", "-o", type=str, default=None, help="Output file name.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Device to run the model.")
    parser.add_argument("--lang", type=str, default="hi", help="Language code for alignment (e.g., 'hi' for Hindi).")
    parser.add_argument("--fresh", action="store_true", help="Run alignment.")
    parser.add_argument("--diarize", action="store_true", help="Run speaker diarization.")

    args = parser.parse_args()
    audio_file = None
    transcript_file = None
    if os.path.isfile(args.data):
        assert args.data.endswith(".mp3"), "Only MP3 audio files are supported."
        audio_file = args.data
        args.data_dir = os.path.dirname(audio_file)
    elif os.path.isdir(args.data):
        args.data_dir = args.data
    else:
        raise ValueError(f"Invalid data path: {args.data}")

    for file in os.listdir(args.data_dir):
        if not audio_file and file.endswith(".mp3"):
            audio_file = os.path.join(args.data_dir, file)
        elif file.endswith(".txt"):
            transcript_file = os.path.join(args.data_dir, file)

    args.out_name = args.out_name or os.path.basename(args.data_dir)

    out_file = os.path.join(args.data_dir, f"{args.out_name}.json")

    assert audio_file is not None, f"No audio file found in {args.data_dir}."
    assert transcript_file is not None, f"No transcript file found in {args.data_dir}."
    audio = load_audio(audio_file)
    
    if os.path.exists(out_file) and not args.fresh:
        alignment_result = read(out_file)
        print(f"Alignment result already exists at {out_file}.")
    else:
        alignment_result = load_and_align(
            audio=audio,
            transcript_file=transcript_file,
            out_file=out_file,
            device=args.device,
            language_code=args.lang
        )
    words = []
    for segment in alignment_result["segments"]:
        words.extend(segment["words"])
    with open(os.path.join(args.data_dir, f"score_transcript.txt"), "w") as f:
        for word in words:
            if "score" in word:
                f.write(f"{word['word']}[{word['score']}] ")
            else:
                f.write(f"{word['word']} ")    
    
    if args.diarize:
        alignment_result = read(out_file)
        result_with_speakers = add_speaker_diarization(audio, alignment_result, device=args.device, min_speakers=1, max_speakers=2)

        diarized_out_file = os.path.join(args.data_dir, f"{args.out_name}_diarized.json")
        write(diarized_out_file, result_with_speakers)
        print(f"Diarized output saved to {diarized_out_file}.")


if __name__ == "__main__":
    main()
