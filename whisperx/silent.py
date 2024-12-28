from pydub import AudioSegment
from pydub.silence import detect_nonsilent

audio = AudioSegment.from_mp3("good_morning.mp3")

nonsilent_ranges = detect_nonsilent(audio, min_silence_len=500, silence_thresh=-40)

# Find the closest natural break within the first 10 minutes
time_ms = 10 * 60 * 1000    # 10 minutes in milliseconds
natural_break = next((end for start, end in nonsilent_ranges if end <= time_ms), time_ms)

first_10_minutes = audio[:natural_break]
first_10_minutes.export("good_morning_10.mp3", format="mp3")
