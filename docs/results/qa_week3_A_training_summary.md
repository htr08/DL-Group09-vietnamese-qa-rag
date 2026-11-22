# Week 3 — A: PhoBERT on clean internal ViQuAD Dev

**Scope and source.** Summary of historical experiment outputs provided in `phobert_3lr_clean_dev_EVIDENCE.zip` (15 files, including each run's history, training summary, run manifest and n-best summary). The source archive is external to the repository; no model weights or training dataset are included in this document.

- Dataset: internal title-held-out Dev, **2,768 questions**, 13 titles, 3,155 features; 1,886 answerable / 882 impossible; zero examples skipped during clean preprocessing, 29 unalignable labels kept for evaluation.
- Base model: `vinai/phobert-base`; 3 epochs; `seed=12`; 3 learning rates: `2e-5`, `3e-5`, `5e-5`.
- Data integrity: no overlapping Train/Dev titles or contexts. `mode=eval` for Dev; no Gold-dependent input/window refinement.
- SHA-256 of clean Dev feature file reported in the provided manifest: `b27046196cd92f0ede419737b155f7f36a6bb1f2c54c563b3bafa31c82d8d3c3`.
- Source fingerprint from reported training manifests: `3e04aedb82c69908f19b1cdcda6499ab51ac497b656ba7826ef90b0b6bd82deb`.

## Results — Dev n-best with Dev-tuned no-answer threshold

| LR | Best epoch | Train-loop Dev F1 (%) | Tuned Dev EM (%) | Tuned Dev F1 (%) | Null threshold |
|---|---:|---:|---:|---:|---:|
| `2e-5` | 3 | 65.59 | 52.42 | 66.39 | -1.483816 |
| `3e-5` | 3 | 67.35 | 53.76 | 67.93 | -0.657978 |
| `5e-5` | 3 | 69.53 | **55.78** | **69.94** | -1.489326 |

**Best on overall Dev F1:** LR `5e-5`, epoch 3. This is a **selection on Dev**, not Public Test. The training-loop F1 at its fixed threshold and postprocessing F1 after threshold tuning must not be conflated.

| LR | Answerable Dev F1 (%) | No-answer Dev EM/F1 (%) |
|---|---:|---:|
| `2e-5` | 67.16 | 64.74 |
| `3e-5` | **70.33** | 62.81 |
| `5e-5` | 68.08 | **73.92** |

## Reproducibility and limitations

- Original training base commit in the provided manifests: `b202d3dc9fb4bcdd02131d213d4b1b8d26f1dde8`. Current CI/reviewer fixes modify code after the recorded runs, so exact historical reproduction requires pinning the original code and dependencies and validating input/checkpoint hashes.
- The evidence ZIP contains metrics and manifests, **not the trained weights**. This Markdown summary alone does not demonstrate successful checkpoint reload or rerun of inference.
- Bootstrap 95% confidence intervals on full ViQuAD Dev have **not** been verified in the evidence ZIP; no such intervals are invented here.
- Historical Gold Leakage audit: old Dev preprocessing changed features for 29/2,768 examples, documented in `docs/results/gold_leakage_audit/`; this is not a measured difference in F1.
- Domain B1 (BM25 + PhoBERT) on the group's 30-question domain Dev remains **provisional until two-round annotation reconciliation**. No domain-Test score is reported here.
- **Public Test not evaluated/tuned** for these results. The plan reserves the single final Test run for Week 5.

See `docs/qa_week2_A_evidence.md`, `docs/qa_week2_A_eda_reproduced.md` and `data/splits/viquad_split_meta.json` for the dataset checks.
