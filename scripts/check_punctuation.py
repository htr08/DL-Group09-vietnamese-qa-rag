from src.preprocessing_qa import normalize_with_mapping

text = "Karl Heinrich Marx (phát âm tiếng Đức: [kaːɐ̯l ˈhaɪnʀɪç ˈmaːɐ̯ks], thường được phiên âm)"
normalized, mapping = normalize_with_mapping(text)

for token in ["[", "kaːɐ̯l", "ˈhaɪnʀɪç", "ˈmaːɐ̯ks", "]", ",", "thường"]:
    target, _ = normalize_with_mapping(token.replace("_", " "))
    print(repr(token), "=>", repr(target), "=>", normalized.find(target))
