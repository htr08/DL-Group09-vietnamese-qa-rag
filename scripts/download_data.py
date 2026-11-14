
import argparse
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_NAME = "taidng/UIT-ViQuAD2.0"
RANDOM_STATE = 42
DEV_RATIO = 0.1

RAW_DIR = ROOT_DIR / "data" / "raw" / "viquad"
SPLIT_DIR = ROOT_DIR / "data" / "splits"

FILES = {
    "train": RAW_DIR / "train.jsonl",
    "dev": RAW_DIR / "dev.jsonl",
    "public_test": RAW_DIR / "public_test.jsonl",
}

TITLE_FILE = SPLIT_DIR / "viquad_dev_titles.txt"

REQUIRED_FIELDS = {
    "id",
    "uit_id",
    "title",
    "context",
    "question",
    "answers",
    "is_impossible",
    "plausible_answers",
}


def validate_records(records, split_name):
    if not records:
        raise ValueError(f"{split_name} is empty.")

    for index, record in enumerate(records):
        missing = REQUIRED_FIELDS - set(record.keys())

        if missing:
            raise ValueError(
                f"{split_name}, record {index}: missing fields {missing}"
            )

        if not isinstance(record["title"], str) or not record["title"].strip():
            raise ValueError(
                f"{split_name}, record {index}: invalid title."
            )

        if not isinstance(record["context"], str):
            raise ValueError(
                f"{split_name}, record {index}: invalid context."
            )

        if not isinstance(record["question"], str):
            raise ValueError(
                f"{split_name}, record {index}: invalid question."
            )

        answers = record["answers"]

        if answers is not None:
            if len(answers["text"]) != len(answers["answer_start"]):
                raise ValueError(
                    f"{split_name}, record {index}: answer fields mismatch."
                )


def write_jsonl(records, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".jsonl.tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    temp_path.replace(output_path)


def read_jsonl(input_path):
    with input_path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def report_statistics(name, records):
    titles = {record["title"] for record in records}

    print(
        f"{name}: {len(records):,} mẫu | "
        f"{len(titles):,} bài viết"
    )


def validate_existing_files():
    for path in list(FILES.values()) + [TITLE_FILE]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}. Use --overwrite to recreate the dataset."
            )

    train = read_jsonl(FILES["train"])
    dev = read_jsonl(FILES["dev"])
    public_test = read_jsonl(FILES["public_test"])

    validate_records(train, "Train")
    validate_records(dev, "Dev")
    validate_records(public_test, "Public Test")

    train_titles = {record["title"] for record in train}
    dev_titles = {record["title"] for record in dev}

    if train_titles & dev_titles:
        raise ValueError("Train and Dev have overlapping titles.")

    saved_titles = {
        line.strip()
        for line in TITLE_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    if saved_titles != dev_titles:
        raise ValueError("Saved Dev titles do not match the Dev dataset.")

    report_statistics("Train", train)
    report_statistics("Dev", dev)
    report_statistics("Public Test", public_test)

    print("Train/Dev title overlap:", len(train_titles & dev_titles))
    print("Existing dataset files passed validation.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate existing raw dataset files.",
    )
    args = parser.parse_args()

    existing_files = [
        path
        for path in list(FILES.values()) + [TITLE_FILE]
        if path.exists()
    ]

    if existing_files and not args.overwrite:
        if len(existing_files) == len(FILES) + 1:
            print("Dataset files already exist. Validating without overwriting.")
            validate_existing_files()
            return

        raise FileExistsError(
            "Some dataset files already exist. "
            "Use --overwrite only if you intend to recreate them."
        )

    print("Loading dataset:", DATASET_NAME)

    dataset = load_dataset(
        DATASET_NAME,
        split=["train", "validation"],
    )

    original_train = dataset[0]
    public_test = dataset[1]

    titles = sorted(set(original_train["title"]))

    rng = random.Random(RANDOM_STATE)
    dev_titles = set(
        rng.sample(titles, round(len(titles) * DEV_RATIO))
    )

    train_records = [
        record
        for record in original_train
        if record["title"] not in dev_titles
    ]

    dev_records = [
        record
        for record in original_train
        if record["title"] in dev_titles
    ]

    public_test_records = list(public_test)

    validate_records(train_records, "Train")
    validate_records(dev_records, "Dev")
    validate_records(public_test_records, "Public Test")

    train_titles = {record["title"] for record in train_records}
    dev_title_set = {record["title"] for record in dev_records}

    if train_titles & dev_title_set:
        raise ValueError("Train and Dev have overlapping titles.")

    if len(train_records) + len(dev_records) != len(original_train):
        raise ValueError("Train/Dev sample count mismatch.")

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    write_jsonl(train_records, FILES["train"])
    write_jsonl(dev_records, FILES["dev"])
    write_jsonl(public_test_records, FILES["public_test"])

    TITLE_FILE.write_text(
        "\n".join(sorted(dev_titles)) + "\n",
        encoding="utf-8",
    )

    report_statistics("Original Train", list(original_train))
    report_statistics("Train", train_records)
    report_statistics("Dev", dev_records)
    report_statistics("Public Test", public_test_records)

    print("Train/Dev title overlap:", len(train_titles & dev_title_set))
    print("Saved dataset files successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
