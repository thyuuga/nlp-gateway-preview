def guess_lang(text: str) -> str:
    """
    Return one of: 'zh', 'ja', 'en'
    Heuristic:
      - if kana present -> ja
      - else if CJK ideographs dominate -> zh
      - else if latin letters dominate -> en
      - else default zh (user's current main testing language)
    """
    if not text:
        return "zh"

    kana = 0
    cjk = 0
    latin = 0
    other = 0

    for ch in text:
        o = ord(ch)
        # Hiragana 3040–309F, Katakana 30A0–30FF, Katakana Phonetic Extensions 31F0–31FF
        if 0x3040 <= o <= 0x309F or 0x30A0 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF:
            kana += 1
        # CJK Unified Ideographs 4E00–9FFF (enough for heuristic)
        elif 0x4E00 <= o <= 0x9FFF:
            cjk += 1
        # Basic Latin letters
        elif (0x41 <= o <= 0x5A) or (0x61 <= o <= 0x7A):
            latin += 1
        elif ch.isspace():
            continue
        else:
            other += 1

    if kana > 0:
        return "ja"

    # choose by dominance (avoid division by 0)
    total = max(1, kana + cjk + latin + other)
    if cjk / total >= 0.25:
        return "zh"
    if latin / total >= 0.30:
        return "en"

    return "zh"
