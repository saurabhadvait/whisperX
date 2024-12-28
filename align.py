import argparse
import os

from pydub import AudioSegment

import whisperx


def get_audio_duration(audio_file):
    audio = AudioSegment.from_file(audio_file)
    return len(audio) / 1000.0  # Duration in seconds

def load_and_align(audio_file, transcript_file, out_file, device="cpu", language_code="hi"):
    audio = whisperx.load_audio(audio_file)
    audio_duration = get_audio_duration(audio_file)

    model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device, model_dir="models/")

    with open(transcript_file, "r") as f:
        transcript = f.read()

    alignment_result = whisperx.align(
        [{"text": transcript, 'start': 0, 'end': audio_duration}],
        model_a, metadata, audio, device, return_char_alignments=False
    )

    with open(out_file, "w") as f:
        print(alignment_result["segments"], file=f)

    print(f"Alignment completed. Output saved to {out_file}.")


def main():
    parser = argparse.ArgumentParser(description="WhisperX Audio Alignment Script")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"], help="Device to run the model.")
    parser.add_argument("--language_code", type=str, default="hi", help="Language code for alignment (e.g., 'hi' for Hindi).")
    parser.add_argument("--out_name", type=str, default="aligned", help="Prefix for the output file.")

    args = parser.parse_args()
    audio_file = None
    transcript_file = None
    for file in os.listdir(args.data_dir):
        if file.endswith(".mp3"):
            audio_file = os.path.join(args.data_dir, file)
        if file.endswith(".txt"):
            transcript_file = os.path.join(args.data_dir, file)
    assert audio_file is not None, f"No audio file found in {args.data_dir}."
    assert transcript_file is not None, f"No transcript file found in {args.data_dir}."
    out_file = os.path.join(args.data_dir, f"{args.out_name}.txt")
    load_and_align(
        audio_file=audio_file,
        transcript_file=transcript_file,
        out_file=out_file,
        device=args.device,
        language_code=args.language_code,
    )


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
