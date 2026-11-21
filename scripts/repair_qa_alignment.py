"""Conservative original-context alignment audit for PhoBERT n-best DEV output.

This is a POSTPROCESSING ABLATION, not a model change and not a full
character-offset solution. Match unique normalized substrings, preserve
exact characters from the original context, and keep original text on failure.
Use only ViQuAD Dev, never Test, to report/tune model selection.

Run at repository root:
python scripts/repair_qa_alignment.py --nbest outputs/week3_nbest/nbest_dev.jsonl \
    --gold data/processed/viquad/dev.jsonl \
    --summary outputs/week3_nbest/summary.json \
    --output-dir outputs/week3_alignment_repair
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Same metric functions used for the prior evaluation: no new normalizer.
# Remove spacing around punctuation only for locating the substring. The
# returned answer always contains the exact original context substring.
_PUNCT = set('.,;:!?()[]{}"“”‘’«»/\\-–—%°')
_QUOTE_EQUIV = {'“': '"', '”': '"', '«': '"', '»': '"', '‘': "'", '’': "'"}


def _graphemes_with_offsets(source: str):
    """Group combining marks with preceding char, preserving source indices."""
    result = []
    for i, ch in enumerate(source):
        if unicodedata.combining(ch) and result:
            raw, start, _ = result[-1]
            result[-1] = (raw + ch, start, i + 1)
        else:
            result.append((ch, i, i + 1))
    return result


def _canonical_with_offsets(source: str, *, drop_accents=False):
    items = []
    for raw, start, end in _graphemes_with_offsets(source):
        val = unicodedata.normalize('NFC', raw).casefold()
        if drop_accents:
            val = ''.join(c for c in unicodedata.normalize('NFD', val)
                          if unicodedata.category(c) != 'Mn')
            val = val.replace('đ', 'd')
        for char in val:
            char = _QUOTE_EQUIV.get(char, char)
            if char.isspace() or char in ('\u00a0', '\u200b'):
                char = ' '
            items.append((char, start, end))

    # Collapse repeated whitespace (retaining boundary offsets).
    collapsed = []
    for char, start, end in items:
        if char == ' ' and collapsed and collapsed[-1][0] == ' ':
            prev_char, prev_start, _ = collapsed[-1]
            collapsed[-1] = (prev_char, prev_start, end)
        else:
            collapsed.append((char, start, end))

    # Ignore whitespace next to punctuation for matching. It is NOT removed
    # from the returned, literal context substring.
    packed = []
    for i, item in enumerate(collapsed):
        ch = item[0]
        if ch == ' ':
            previous = collapsed[i - 1][0] if i else ''
            following = collapsed[i + 1][0] if i + 1 < len(collapsed) else ''
            if previous in _PUNCT or following in _PUNCT:
                continue
        packed.append(item)
    # Trim outer spaces from canonical representation.
    while packed and packed[0][0] == ' ':
        packed.pop(0)
    while packed and packed[-1][0] == ' ':
        packed.pop()
    return ''.join(ch for ch, _, _ in packed), [(s, e) for _, s, e in packed]


def _unique_match(candidate: str, context: str, *, drop_accents=False):
    hay, offsets = _canonical_with_offsets(context, drop_accents=drop_accents)
    needle, _ = _canonical_with_offsets(candidate.replace('_', ' '), drop_accents=drop_accents)
    if not needle:
        return None
    found = []
    at = 0
    while (index := hay.find(needle, at)) != -1:
        right = index + len(needle)
        # Prevent matching within longer words.
        left_ok = index == 0 or not hay[index - 1].isalnum() or not needle[0].isalnum()
        right_ok = right == len(hay) or not hay[right].isalnum() or not needle[-1].isalnum()
        if left_ok and right_ok:
            start = offsets[index][0]
            end = offsets[right - 1][1]
            found.append(context[start:end])
            if len(found) >= 2:
                return None  # ambiguous: do not guess which occurrence
        at = index + 1
    return found[0] if len(found) == 1 else None


def recover_original_span(candidate: str, context: str):
    """Return (text, category); never manufacture text not in context.

    Unaligned text is retained explicitly so metric effects can be audited.
    The accent-fold fallback is only attempted for multi-word long strings.
    """
    if not candidate or not context:
        return candidate, 'empty_or_missing'
    normalized = ' '.join(candidate.replace('_', ' ').split())
    escaped = r'\s+'.join(re.escape(x) for x in normalized.split(' '))
    match = re.search(escaped, context, flags=re.IGNORECASE)
    if match:
        return match.group(0), 'exact_whitespace'
    repaired = _unique_match(candidate, context, drop_accents=False)
    if repaired is not None:
        return repaired, 'punctuation_whitespace'
    if len(candidate.strip()) >= 9 and len(candidate.split()) >= 2:
        repaired = _unique_match(candidate, context, drop_accents=True)
        if repaired is not None:
            return repaired, 'accent_fold_unique'
    return candidate, 'unaligned'


def read_jsonl(path):
    with open(path, encoding='utf-8-sig') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def score(rows, answers_by_id, threshold):
    from src.qa_metrics import exact_match_score, f1_score
    em = f1 = 0.0
    per_class = {'answerable': {'n': 0, 'em': 0.0, 'f1': 0.0},
                 'impossible': {'n': 0, 'em': 0.0, 'f1': 0.0}}
    for row in rows:
        gold = answers_by_id[row['id']]
        is_null = bool(gold['is_impossible'])
        refs = [''] if is_null else gold['answers']['text']
        group = per_class['impossible' if is_null else 'answerable']
        pred = row['prediction']
        this_em = float(exact_match_score(pred, refs))
        this_f1 = float(f1_score(pred, refs))
        em += this_em
        f1 += this_f1
        group['n'] += 1
        group['em'] += this_em
        group['f1'] += this_f1
    for group in per_class.values():
        if group['n']:
            group['em'] /= group['n']
            group['f1'] /= group['n']
    return {'em': em / len(rows), 'f1': f1 / len(rows), 'per_category': per_class}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nbest', type=Path, default=Path('outputs/week3_nbest/nbest_dev.jsonl'))
    parser.add_argument('--gold', type=Path, default=Path('data/processed/viquad/dev.jsonl'))
    parser.add_argument('--summary', type=Path, default=Path('outputs/week3_nbest/summary.json'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/week3_alignment_repair'))
    args = parser.parse_args()
    inputs = [args.nbest.resolve(), args.gold.resolve(), args.summary.resolve()]
    if any(not p.is_file() for p in inputs):
        raise FileNotFoundError(f'Missing input: {[str(p) for p in inputs if not p.exists()]}')
    nbest = read_jsonl(inputs[0])
    gold = read_jsonl(inputs[1])
    summary = json.loads(inputs[2].read_text(encoding='utf-8'))
    nbest_by_id = {r['id']: r for r in nbest}
    gold_by_id = {r['id']: r for r in gold}
    if len(gold) != 2768 or len({g.get('title') for g in gold}) != 13 or len(nbest) != 2768:
        raise ValueError('Expected complete 13-title ViQuAD Dev of 2768 questions')
    if len(nbest_by_id) != 2768 or len(gold_by_id) != 2768 or set(nbest_by_id) != set(gold_by_id):
        raise ValueError('Duplicate or mismatched IDs')
    if set(summary['dev_titles']) != {g['title'] for g in gold}:
        raise ValueError('Summary Dev titles differ from gold')
    threshold = float(summary['tuned_on_dev']['null_threshold'])
    counts = Counter()
    original, modified, changes = [], [], []
    for item in nbest:
        qid = item['id']
        rec = gold_by_id[qid]
        recovered, method = recover_original_span(item['best_answer'], rec['context'])
        counts[method] += 1
        empty = float(item['score_diff']) > threshold
        old_prediction = '' if empty else item['best_answer']
        new_prediction = '' if empty else recovered
        original.append({'id': qid, 'prediction': old_prediction})
        modified.append({'id': qid, 'prediction': new_prediction})
        if old_prediction != new_prediction:
            changes.append({'id': qid, 'category': 'impossible' if rec['is_impossible'] else 'answerable',
                            'method': method, 'old': old_prediction, 'new': new_prediction,
                            'score_diff': item['score_diff']})
    old_scores = score(original, gold_by_id, threshold)
    # This must reproduce the prior Dev baseline before we attribute *any*
    # difference to postprocessing. Fail closed on metadata/metric drift.
    expected = summary['tuned_on_dev']
    if abs(old_scores['em'] - float(expected['em'])) > 1e-6 or abs(old_scores['f1'] - float(expected['f1'])) > 1e-6:
        raise ValueError('Recomputed ORIGINAL Dev EM/F1 differs from summary; verify metric/inputs before comparing')
    new_scores = score(modified, gold_by_id, threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'repaired_dev_predictions.jsonl').open('w', encoding='utf-8') as f:
        for r in modified:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with (args.output_dir / 'changed_predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'category', 'method', 'old', 'new', 'score_diff'])
        writer.writeheader()
        writer.writerows(changes)
    report = {
        'note': 'DEV-only fixed-threshold output repair, not a change to model logits or training.',
        'threshold_fixed': threshold, 'n_questions': len(nbest),
        'alignment_methods_all_best_spans': dict(counts),
        'changed_final_predictions': len(changes),
        'original_dev_scores_recomputed': old_scores,
        'repaired_dev_scores': new_scores,
        'delta_f1_percentage_points': round(100 * (new_scores['f1'] - old_scores['f1']), 5),
        'delta_em_percentage_points': round(100 * (new_scores['em'] - old_scores['em']), 5),
    }
    (args.output_dir / 'repair_summary.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print('Output:', args.output_dir.resolve())


if __name__ == '__main__':
    main()
