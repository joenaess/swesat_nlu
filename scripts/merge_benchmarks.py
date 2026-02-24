import json
import os
import glob
from datasets import load_dataset
from prompts import zero_shot_prompts


def get_system_prompt(options_count):
    if options_count == 0:
        return ""
    letters = ["A", "B", "C", "D", "E"][:options_count]
    prompt = "Du är en expert på att skriva högskoleprovet.\n\n"
    for letter in letters:
        prompt += f"- Om det rätta svaret är svarsalternativ {letter}, skriv bara ut {letter}\n"
    if len(letters) > 1:
        prompt += f"- Skriv bara ut {', '.join(letters[:-1])} eller {letters[-1]} - inget annat\n"
    elif len(letters) == 1:
        prompt += f"- Skriv bara ut {letters[0]} - inget annat\n"
    return prompt


def construct_prompt(subsection, question, options):
    prompt_str = f"\n{subsection}\n"
    instruction = zero_shot_prompts.get(subsection, "")
    if instruction:
        prompt_str += f"{instruction}\n\n"

    prompt_str += f"{question}\n\n"
    for letter, opt_text in options.items():
        if opt_text:
            prompt_str += f"{letter}: {opt_text}\n"
    prompt_str += "\nSvar:\n"
    return prompt_str


def get_answer_key(facit_data, filename, question_number):
    provpass_match = None
    if "provpass-1" in filename:
        provpass_match = "provpass-1"
    elif "provpass-2" in filename:
        provpass_match = "provpass-2"
    elif "provpass-3" in filename:
        provpass_match = "provpass-3"
    elif "provpass-4" in filename:
        provpass_match = "provpass-4"
    elif "provpass-5" in filename:
        provpass_match = "provpass-5"

    if not provpass_match:
        return None

    return facit_data.get(provpass_match, {}).get(str(question_number))


def load_swesat():
    unified_items = []
    base_dir = "exams"

    for date_dir in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, date_dir)
        if not os.path.isdir(dir_path):
            continue

        facit_files = glob.glob(os.path.join(dir_path, "*facit*.json"))
        facit_data = {}
        if facit_files:
            try:
                with open(facit_files[0], "r", encoding="utf-8") as f:
                    facit_data = json.load(f)
            except Exception as e:
                print(f"Failed to load facit {facit_files[0]}: {e}")

        qa_files = [
            f
            for f in glob.glob(os.path.join(dir_path, "*.json"))
            if "facit" not in f.lower()
        ]
        for qa_file in qa_files:
            try:
                with open(qa_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                continue

            for item in data:
                # Skip visual questions
                if (
                    item.get("is_accompanied_with_visual") == "yes"
                    or "visual required" in item.get("question", "").lower()
                ):
                    continue

                q_num = item.get("question_number")
                ans_key = get_answer_key(facit_data, os.path.basename(qa_file), q_num)

                options_dict = {
                    "A": item.get("answers", {}).get("a", ""),
                    "B": item.get("answers", {}).get("b", ""),
                    "C": item.get("answers", {}).get("c", ""),
                    "D": item.get("answers", {}).get("d", ""),
                    "E": item.get("answers", {}).get("e", ""),
                }
                valid_options = {k: v for k, v in options_dict.items() if v}

                if not valid_options:
                    continue

                subsection = item.get("question_type", "")
                system_prompt = get_system_prompt(len(valid_options))
                user_prompt = construct_prompt(
                    subsection, item.get("question", ""), valid_options
                )

                unified_item = {
                    "uid": f"{date_dir}_{os.path.basename(qa_file).split('.')[0]}_{subsection}_q-{q_num}",
                    "test_id": date_dir,
                    "section": os.path.basename(qa_file).split(".")[0],
                    "subsection": subsection,
                    "question_id": q_num,
                    "question_resource": None,
                    "question": item.get("question", ""),
                    "option_a": options_dict["A"],
                    "option_b": options_dict["B"],
                    "option_c": options_dict["C"],
                    "option_d": options_dict["D"],
                    "option_e": options_dict["E"],
                    "system_prompt": system_prompt,
                    "prompt": user_prompt,
                    "answer": ans_key,
                    "source": "swesat",
                }
                unified_items.append(unified_item)

    return unified_items


def load_skolprov():
    print("Loading Swedish Skolprov from HF...")
    skolprov_ds = load_dataset("Ekgren/swedish_skolprov", "all")
    split_name = "train" if "train" in skolprov_ds else list(skolprov_ds.keys())[0]

    unified_items = []

    for item in skolprov_ds[split_name]:
        if item.get("question_resource"):
            continue

        unified_item = {
            "uid": item.get("uid", ""),
            "test_id": item.get("test_id", ""),
            "section": item.get("section", ""),
            "subsection": item.get("subsection", ""),
            "question_id": item.get("question_id", ""),
            "question_resource": item.get("question_resource", ""),
            "question": item.get("question", ""),
            "option_a": item.get("option_a", ""),
            "option_b": item.get("option_b", ""),
            "option_c": item.get("option_c", ""),
            "option_d": item.get("option_d", ""),
            "option_e": item.get("option_e", ""),
            "system_prompt": item.get("system_prompt", ""),
            "prompt": item.get("prompt", ""),
            "answer": item.get("answer", ""),
            "source": "skolprov",
        }
        unified_items.append(unified_item)

    return unified_items


def merge():
    print("Parsing local swesat exams...")
    swesat_data = load_swesat()
    print(f"Loaded {len(swesat_data)} valid text-only Swesat questions.")

    skolprov_data = load_skolprov()
    print(f"Loaded {len(skolprov_data)} valid text-only Skolprov questions.")

    combined = swesat_data + skolprov_data

    # Deduplicate combined items by their prompt and answer
    seen = set()
    deduped = []
    for item in combined:
        sig = (item.get("prompt", ""), item.get("answer", ""))
        if sig not in seen:
            seen.add(sig)
            deduped.append(item)

    print(f"Removed {len(combined) - len(deduped)} duplicate questions from overlap.")
    combined = deduped

    output_file = "merged_benchmark.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in combined:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        f"\nSuccessfully merged {len(combined)} total questions with full prompt annotations into {output_file}."
    )


if __name__ == "__main__":
    merge()
