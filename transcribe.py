import base64
import json
import os
import time
from argparse import ArgumentParser

from openai import OpenAI


def transcribe_audio(audio_path: str, config_path: str) -> str:
    assert audio_path.endswith(".mp3"), "Only MP3 audio files are supported."
    with open(config_path, "r") as f:
        config = json.load(f)
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


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument(
        "--audio_path",
        type=str,
        required=True,
        help="Path to the audio file.",
    )
    args.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Path to save the transcription.",
    )
    args.add_argument(
        "--config",
        type=str,
        default="asr_final",
        help="Name of the configuration file.",
    )
    args = args.parse_args()
    audio_path = args.audio_path
    if not args.out_path:
        out_path = audio_path.replace(".mp3", "_transcript.txt")
    start_time = time.time()
    transcript = transcribe_audio(audio_path, config_path=f"configs/{args.config}.json").strip("...")
    time_taken = time.time() - start_time
    with open(out_path, "w") as f:
        f.write(transcript)
    print(f"Transcription saved to {out_path}, took {time_taken:.2f} seconds.")
