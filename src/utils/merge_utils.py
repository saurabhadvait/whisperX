import re
from typing import Dict, List, Tuple

from src.utils import write


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
    text = re.sub(r'[\u0964\u0965]', ' ', text)    # hindi full stop so that it doesn't match the below pattern
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
        "end_in_merged": f[sf + ml - 1]["end"],
        "match_length": ml
    }

if __name__ == "__main__":
    # first_transcript = r"""कहानियां खूब सुना देगा,मनोरंजन की बात
    # बहुत बता देगा, कह देगा-हमारी तरफ ऐसा माना जाता है, हमारी तरफ
    # फलानी नदी, फलाने वृक्ष की मान्यता है, ये सब वो खूब बता देगा।

    # ----------------------------------------

    # Speaker 1: कुछ रस्मो रिवाज बता देगा, कुछ खानपान की चीजें कह देगा
    # । कुछ ऐसी बातें; कह

    # देगा: जिसको।। हम संस्कृति-blah या culture कह सकते हैं। ल
    # किन धार्मिकता के केंद्र में बंधनों """

    # second_transcript = r"""kyo;हम-संस्कृति blah या culture     कह सकते हैं। लेकिन      धार्मिकता के केंद्र में kya hoga बंधनों ka"""
    first_transcript = "Speaker 2: देखिए, United Nations Framework Convention on Climate Change है। इसमें बीच-बीच में कुछ बातें बताऊंगा जो सूचना जैसी हैं, उसको ले लीजिएगा, उसी से बात पूरी समझ में आएगी। तो उसके तहत हर साल लोग मिलते हैं, जैसे अभी मिले थे न बाकू में। सारे देश वहाँ पर बैठक करते हैं, उसे बोलते हैं Conference of Parties, COP। तो जो पंद्रहवीं COP हुई थी पेरिस में, वहाँ पेरिस समझौता हुआ। पेरिस समझौता। और पेरिस समझौते ने कहा 2015 में कि जितना अभी carbon emission का स्तर है, इसको 45% घटाना होगा 2030 तक और net zero पर आना होगा 2050 तक। क्यों आना होगा? ये आँकड़े क्यों निर्धारित करे गए? इसलिए निर्धारित करे गए ताकि किसी भी तरीके से जो तापमान में वृद्धि है, global temperature rise है, उसको 2 डिग्री centigrade से नीचे रोका जा सके। ये उद्देश्य था। कहा कि डेढ़ डिग्री हम चाहते हैं, पर डेढ़ डिग्री न भी हो पाए, तो 2 डिग्री से ज्यादा आगे नहीं जाना चाहिए temperature rise। ये हमने 2015 में समझा।"
    second_transcript = "Speaker 1: तापमान में वृद्धि है, global temperature rise है, उसको 2 degree centigrade से नीचे रोका जा सके, ये उद्देश्य था। कहा कि डेढ़ degree हम चाहते हैं, पर डेढ़ degree न भी हो पाए, तो 2 degree से ज़्यादा आगे नहीं जाना चाहिए temperature rise। ये हमने 2015 में समझौता किया था, 195 देश मिलकर बैठे थे। और सबको ये बात समझ में आई थी कि 2 degree पर इसको रोकना बहुत ज़रूरी है। तो लक्ष्य था temperature rise को रोकना, डेढ़ degree का लक्ष्य निर्धारित किया गया। और उसके लिए जो रास्ता सुझाया गया वो ये था कि 2030 तक अपने emissions को लगभग आधा कर दो और 2050 तक net zero पर आ जाओ।"
    ft = extract_words_with_positions(first_transcript)
    ft = [w["token"] for w in ft]
    st = extract_words_with_positions(second_transcript)
    st = [w["token"] for w in st]
    sf, ss, ml = find_longest_consecutive_overlap(ft, st)
    write("out.txt", f"{ft}\n\n{st}\n\nMatch length={ml}\n\n{ft[sf:sf+ml]}")
