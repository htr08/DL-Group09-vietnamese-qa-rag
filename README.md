# Vietnamese QA: open-source RAG and extractive reading comprehension

Course project (Deep Learning, Group 9, Topic 10). The system replaces the commercial-API components of an earlier RAG system (Gemini embedding and Gemini 2.5 Flash) with BGE-M3 and Qwen2.5-7B-Instruct, and adds an extractive QA module (PhoBERT, with XLM-R as a comparison) evaluated on UIT-ViQuAD 2.0 and on a self-annotated set of 100 domain questions.

Status: project setup. Results will be added after the experiments in `docs/charter.md` are completed.

## Repository layout

| Path | Content |
|---|---|
| `src/` | Library code (QA preprocessing, training, retrieval, generation, metrics, pipeline) |
| `scripts/` | Entry points: data download, index building, evaluation |
| `tests/` | Unit tests (run by CI on every pull request) |
| `notebooks/` | EDA, analysis and the Kaggle smoke test (`00_kaggle_qwen_smoke_test.ipynb`) |
| `data/raw/` | Downloaded datasets (git-ignored) |
| `data/domain/` | Application-domain PDFs (Samsung Galaxy S24 specification sheets) |
| `data/gold/` | Annotated 100-question set (dev/test) |
| `data/splits/` | Split indices and seeds |
| `kaggle/smoke_test_dataset/` | Small dataset used by the Kaggle smoke test |
| `app/` | Demo application |
| `docs/` | Project plan, charter, AI-usage declaration and reuse-vs-new documentation |
| `report/` | LaTeX reports in the ACL template: `proposal/`, `milestone/`, `final/` |

The code of the base system (AI Marketing Assistant) is not copied into this repository. Its source repository is linked in `docs/reuse_vs_new.md`.

## Setup

Python 3.12 and, for VnCoreNLP word segmentation, a JDK (Java 8 or later) on `PATH`.

```bash
python -m venv venv
# Windows:      venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate
pip install -r requirements.txt
```

`torch` is not pinned in `requirements.txt`; install the build matching the machine (see the comment at the top of that file).

## Experiment tracking (Weights & Biases)

Every run that produces numbers for the report (training, evaluation, index builds) is logged to the shared W&B project: team `nhom09-vi-qa-rag`, project `vi-qa-rag`. Run names follow `<role>_<model>_<config>_seed<k>`.

```bash
pip install wandb==0.31.0
wandb login            # paste your own key from https://wandb.ai/authorize; never commit it
```

On Kaggle, add the key as a secret named `WANDB_API_KEY` (Add-ons, Secrets) and export it in the notebook:

```python
import os
from kaggle_secrets import UserSecretsClient
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
```

Use the shared helper so entity, project and commit hash stay consistent:

```python
from src.tracking import init_run
run = init_run("A_phobert_lr3e-5_seed42", job_type="train", config={"lr": 3e-5})
```

Set `WANDB_MODE=disabled` to switch logging off (CI, quick local checks).

## Tests

```bash
pip install -r requirements-ci.txt
pytest
```

## Self-implemented vs. reused components

To be completed as the project proceeds.

| Component | Self-implemented | Reused (source) |
|---|---|---|
| (to be filled) | | |
