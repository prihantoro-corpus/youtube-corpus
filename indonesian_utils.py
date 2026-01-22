import re

PRONOMINAL_SE = {
    "semua", "segala", "sesuatu", "seseorang",
    "sebagian", "sejumlah", "sekelompok"
}

BLOCK_KU_PREFIX = {
    "kuliah", "kuning", "kurang"
}

BLOCK_KU_MU_SUFFIX = {
    "aku", "kamu", "ilmu", "tamu"
}


def tokenize_word(word):
    prefix_tokens = []
    suffix_tokens = []
    W = word
    w_lower = word.lower()

    # RULE 0 — very short
    if len(W) <= 3:
        return [W]

    # RULE 1 — proklitik ku-
    if w_lower.startswith("ku"):
        remainder_lower = w_lower[2:]
        if len(remainder_lower) >= 4 and w_lower not in BLOCK_KU_PREFIX:
            prefix_tokens.append(W[:2])
            W = W[2:]
            w_lower = remainder_lower

    # RULE 2 — enklitik -nya
    if w_lower.endswith("nya") and len(W) > 5:
        if w_lower.startswith("se"):
            base_lower = w_lower[:-3]
            if base_lower in PRONOMINAL_SE:
                suffix = W[-3:]
                W = W[:-3]
                suffix_tokens.insert(0, suffix)
                w_lower = base_lower
        else:
            suffix_tokens.insert(0, W[-3:])
            W = W[:-3]
            w_lower = w_lower[:-3]

    # RULE 3 — enklitik -ku / -mu
    if w_lower.endswith("ku") or w_lower.endswith("mu"):
        suffix = W[-2:]
        base_lower = w_lower[:-2]
        if len(base_lower) >= 4 and w_lower not in BLOCK_KU_MU_SUFFIX:
            W = W[:-2]
            w_lower = base_lower
            suffix_tokens.insert(0, suffix)

    return prefix_tokens + [W] + suffix_tokens


def tokenize_text(text):
    # kata ATAU simbol
    raw_tokens = re.findall(r"\w+|[^\w\s]", text)
    result = []
    for tok in raw_tokens:
        if tok.isalnum():
            result.extend(tokenize_word(tok))
        else:
            result.append(tok)
    return result
