import gc
import sys

import torch

import whisperx
from utils import *

pass
device = "cpu" 
audio_file = sys.argv[1]
out_file = f"data/whisper_{audio_file.split('/')[-1].split('.')[0]}.json"
batch_size = 4 # reduce if low on GPU mem
compute_type = "float32"   #"float16" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
# model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# save model to local path (optional)
model = whisperx.load_model("large-v3", device, compute_type=compute_type, download_root="models/")

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
write(out_file, result)
breakpoint()
# print(result["segments"]) # before alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code="hi", device=device, model_dir="models/")
transcript = open("data/mini_bakriid_trim.txt").read()
result = whisperx.align([{"text": transcript, 'start': 0, 'end': 120}], model_a, metadata, audio, device, return_char_alignments=False)

print(result["segments"], file=open(out_file, "w")) # after alignment

# # delete model if low on GPU resources
# # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

 # 3. Assign speaker labels
diarize_model = whisperx.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)

# add min/max number of speakers if known
diarize_segments = diarize_model(audio)
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

result = whisperx.assign_word_speakers(diarize_segments, result)
print(diarize_segments)
print(result["segments"]) # segments are now assigned speaker IDs