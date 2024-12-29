import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from pydub import AudioSegment


def get_audio_duration(audio_file):
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
    print(f"Reading {file_path}") if verbose else None
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
    return data
