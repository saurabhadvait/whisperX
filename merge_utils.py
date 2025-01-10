import re
from typing import Dict, List, Tuple


def find_longest_consecutive_overlap(a: List[str], b: List[str]) -> Tuple[int, int, int]:
    la, lb = len(a), len(b)
    dp = [[0]*(lb+1) for _ in range(la+1)]
    max_len = 0
    end_a = end_b = 0

    for i in range(1, la+1):
        for j in range(1, lb+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_a, end_b = i, j
            else:
                dp[i][j] = 0

    start_a = end_a - max_len
    start_b = end_b - max_len
    return start_a, start_b, max_len


def extract_words_with_positions(text: str) -> List[Dict[str, int]]:
    pattern = r'[A-Za-z0-9\u0900-\u097f]+'
    return [
        {
            "token": m.group().lower(),
            "start": m.start(),
            "end": m.end()
        }   # token = text[m.start():m.end()]
        for m in re.finditer(pattern, text)
    ]


def merge(first_tr: str, second_tr: str) -> str:
    f = extract_words_with_positions(first_tr)
    s = extract_words_with_positions(second_tr)

    tokens_f = [w["token"] for w in f]
    tokens_s = [w["token"] for w in s]
    sf, ss, ml = find_longest_consecutive_overlap(tokens_f, tokens_s)

    print(f"{ml} Matching tokens:{tokens_f[sf:sf+ml]}")
    if ml == 0 and len(tokens_f) > 0 and len(tokens_s) > 0:
        raise ValueError(f"No overlap found between the two transcripts: $${first_tr}\n$$\n{second_tr}")

    return {
        "merged": first_tr[: f[sf + ml - 1]["end"]] + second_tr[s[ss + ml - 1]["end"] :],
        "end_of_second": s[ss + ml - 1]["end"],  # start is inclusive and end is exclusive
        "end_in_merged": f[sf + ml - 1]["end"]
    }

if __name__ == "__main__":
    first_transcript = r"""कहानियां खूब सुना देगा,मनोरंजन की बात
    बहुत बता देगा, कह देगा-हमारी तरफ ऐसा माना जाता है, हमारी तरफ
    फलानी नदी, फलाने वृक्ष की मान्यता है, ये सब वो खूब बता देगा।

    ----------------------------------------

    Speaker 1: कुछ रस्मो रिवाज बता देगा, कुछ खानपान की चीजें कह देगा
    । कुछ ऐसी बातें; कह

    देगा: जिसको।। हम संस्कृति-blah या culture कह सकते हैं। ल
    किन धार्मिकता के केंद्र में बंधनों """

    second_transcript = r"""kyo;हम-संस्कृति blah या culture कह सकते हैं। लेकिन धार्मिकता के केंद्र में kya hoga बंधनों ka"""
    print(f"Merged Transcript:{merge(first_transcript, second_transcript)}")
