import os
"""Small regression tests for analyze_clean_dev_errors.py; no model needed."""
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'analyze_clean_dev_errors.py'


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def run_case(tmp_path, *, eval_split='ViQuAD title-held-out Dev', bad_answers=False):
    gold = [
        {'id': 'a1', 'title': 'T', 'question': 'Q1', 'context': 'coffee',
         'is_impossible': False, 'answers': None if bad_answers else {'text': ['coffee']}},
        {'id': 'a2', 'title': 'T', 'question': 'Q2', 'context': 'alpha',
         'is_impossible': False, 'answers': {'text': ['alpha']}},
        {'id': 'a3', 'title': 'T', 'question': 'Q3', 'context': 'context',
         'is_impossible': True, 'answers': None},
    ]
    predictions = [
        {'id': 'a1', 'prediction': '!!!'},
        {'id': 'a2', 'prediction': 'wrong'},
        {'id': 'a3', 'prediction': 'incorrect'},
    ]
    summary = {
        'eval_split': eval_split,
        'dev_questions': 3,
        'checkpoint': str(tmp_path / 'missing-checkpoint'),
        'tuned_on_dev': {'em': 0.0, 'f1': 0.0, 'null_threshold': 0.0},
    }
    gold_path = tmp_path / 'dev.jsonl'
    pred_path = tmp_path / 'latest_dev_predictions.jsonl'
    summary_path = tmp_path / 'summary.json'
    out_dir = tmp_path / 'out'
    write_jsonl(gold_path, gold)
    write_jsonl(pred_path, predictions)
    summary_path.write_text(json.dumps(summary), encoding='utf-8')
    result = subprocess.run(
        [sys.executable, str(SCRIPT), '--gold', str(gold_path),
         '--predictions', str(pred_path), '--summary', str(summary_path),
         '--out-dir', str(out_dir), '--expected-count', '3', '--sample-size', '3'],
        text=True, encoding='utf-8', capture_output=True,
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
        cwd=str(ROOT),
    )
    return result, out_dir


def test_dev_latest_name_punctuation_null_and_missing_checkpoint(tmp_path):
    result, out_dir = run_case(tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'WARNING: QA head checkpoint missing' in result.stderr
    assert 'WARNING: backbone checkpoint missing' in result.stderr
    report = json.loads((out_dir / 'audit_summary.json').read_text(encoding='utf-8'))
    assert report['sha256_qa_head'] is None
    assert report['sha256_backbone'] is None
    assert report['errors_by_em'] == {'wrong_span': 1, 'false_negative': 1, 'false_positive': 1}
    with (out_dir / 'all_dev_errors.csv').open(encoding='utf-8-sig', newline='') as f:
        labels = {row['id']: row['error_type'] for row in csv.DictReader(f)}
    assert labels == {'a1': 'false_negative', 'a2': 'wrong_span', 'a3': 'false_positive'}


def test_rejects_public_test_from_summary(tmp_path):
    result, out_dir = run_case(tmp_path, eval_split='ViQuAD public Test')
    assert result.returncode != 0
    assert 'summary' in result.stderr
    assert not (out_dir / 'audit_summary.json').exists()


def test_null_answers_for_answerable_gold_rejected_cleanly(tmp_path):
    result, _ = run_case(tmp_path, bad_answers=True)
    assert result.returncode != 0
    assert 'Gold answerable' in result.stderr
    assert 'AttributeError' not in result.stderr
