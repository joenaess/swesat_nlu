import json
import glob
import os
import re
from datasets import load_dataset


def normalize_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[\$\_\{\}\\,]", " ", text)
    text = re.sub(r"[^a-zåäö0-9]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_local_swesat(base_dir):
    questions = []
    for filepath in glob.glob(
        os.path.join(base_dir, "exams", "**", "*.json"), recursive=True
    ):
        if "facit" in filepath.lower():
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    q_text = item.get("question", "")
                    if (
                        item.get("is_accompanied_with_visual") == "yes"
                        or "visual required" in q_text.lower()
                    ):
                        continue

                    if q_text:
                        item["source_file"] = filepath
                        questions.append(item)
        except Exception as e:
            pass
    return questions


def check_overlaps():
    print("Loading Swedish Skolprov from HF...")
    skolprov_ds = load_dataset("Ekgren/swedish_skolprov", "all")
    skolprov_questions = []

    split_name = "train" if "train" in skolprov_ds else list(skolprov_ds.keys())[0]

    unique_test_ids = set()

    for item in skolprov_ds[split_name]:
        q_text = item.get("question", "")
        if item.get("test_id"):
            unique_test_ids.add(item["test_id"])

        if item.get("question_resource"):
            continue

        if q_text:
            skolprov_questions.append(
                {
                    "id": item.get("uid", ""),
                    "test_id": item.get("test_id", ""),
                    "text": q_text,
                    "normalized": normalize_text(q_text),
                }
            )

    print("\nSkolprov Unique Test IDs:")
    print(unique_test_ids)

    print("\nLoading local Swesat data...")
    swesat_questions = load_local_swesat(".")

    skolprov_norm_set = {
        q["normalized"] for q in skolprov_questions if len(q["normalized"]) > 10
    }

    shared_count = 0

    for sq in swesat_questions:
        norm_q = normalize_text(sq["question"])
        if len(norm_q) <= 10:
            continue

        if norm_q in skolprov_norm_set:
            shared_count += 1

    print(f"\nResults:")
    print(f"Total Skolprov questions (text only): {len(skolprov_questions)}")
    print(f"Total Swesat questions (text only): {len(swesat_questions)}")
    print(f"Shared questions (Overlap): {shared_count}")


if __name__ == "__main__":
    check_overlaps()
