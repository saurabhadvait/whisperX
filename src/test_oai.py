import base64

from openai import OpenAI

# data_dir = "/Users/saurabhpurohit/Documents/whisperX/data/marr"
data_dir = "/Users/saurabhpurohit/Documents/whisperX/data/hindi_interview"

# with open(f"{data_dir}/marriage_2m.mp3", "rb") as f:
with open(f"{data_dir}/2m.mp3", "rb") as f:
    audio_bytes = f.read()
    audio_data_base64 = base64.b64encode(audio_bytes).decode("utf-8")

client = OpenAI()
response = client.chat.completions.create(
  model="gpt-4o-audio-preview-2024-12-17",
  messages=[
    {
      "role": "system",
      "content": [
        {
          "text": "You are an advanced transcriptor. You must output the complete and accurate speaker-diarized transcript of the provided audio with words in their respective original languages.",
          "type": "text"
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
            "type": "text",
            "text": "Output the speaker-separated transcript of this audio with words in their respective original languages."
        },
        {
          "type": "input_audio",
          "input_audio": {
            "data": audio_data_base64,
            "format": "mp3"
          }
        }
      ]
    }
  ],
  modalities=["text"],
  temperature=0,
  max_completion_tokens=16384,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)

print(response.to_json(indent=2), file=open(f"{data_dir}/final.json", "w"))
