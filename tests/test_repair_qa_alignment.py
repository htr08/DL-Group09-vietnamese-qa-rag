from scripts.repair_qa_alignment import recover_original_span


def test_accent_position():
    text, method = recover_original_span('Bộ Văn hoá', 'Đây là Bộ Văn hóa của tỉnh.')
    assert text == 'Bộ Văn hóa'
    assert method == 'accent_fold_unique'


def test_punctuation_spaces():
    text, method = recover_original_span('Cục Bảo vệ Môi trường Mỹ ( EPA )', 'Cục Bảo vệ Môi trường Mỹ (EPA)')
    assert text == 'Cục Bảo vệ Môi trường Mỹ (EPA)'
    assert method == 'punctuation_whitespace'


def test_no_match_kept():
    text, method = recover_original_span('máy tính học sâu không có', 'Hôm nay trời đẹp.')
    assert text == 'máy tính học sâu không có'
    assert method == 'unaligned'


def test_ambiguous_normalized_match_not_guessed():
    text, method = recover_original_span('Bộ Văn hoá', 'Bộ Văn hóa, và một Bộ Văn hóa khác.')
    assert text == 'Bộ Văn hoá'
    assert method == 'unaligned'


def test_preserves_source_original_characters():
    source = 'Địa điểm là Nghị định số 9-BNV/NC/NP tại đây.'
    text, method = recover_original_span('Nghị định số 9 - BNV / NC / NP', source)
    assert text == 'Nghị định số 9-BNV/NC/NP'
    assert method == 'punctuation_whitespace'
