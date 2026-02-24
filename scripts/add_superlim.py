import json
import os
from datasets import load_dataset, get_dataset_config_names


def fix_row(row):
    """
    Some SuperLim-2 JSONL files were incorrectly uploaded to HF Hub as TSV,
    so the entire JSON string is in the first column or a seemingly random column.
    This function parses the string if it's a valid JSON object.
    """
    for k, v in row.items():
        if isinstance(v, str) and v.strip().startswith("{") and v.strip().endswith("}"):
            try:
                parsed = json.loads(v)
                # If it successfully parses as a dict, return it instead.
                if isinstance(parsed, dict):
                    return parsed
            except:
                pass
    return row


def map_superlim():
    configs = get_dataset_config_names("sbx/superlim-2")
    unified_items = []

    for conf in configs:
        print(f"Loading {conf}...")
        try:
            ds = load_dataset("sbx/superlim-2", conf, split="train")
        except Exception:
            try:
                ds = load_dataset("sbx/superlim-2", conf, split="test")
            except Exception:
                print(f"  -> Skipped {conf}, could not load train or test split.")
                continue

        count = 0
        for i, row in enumerate(ds):
            # Parse broken schemas
            row = fix_row(row)

            LABEL_MAP = {
                "incorrect": "Inkorrekt",
                "correct": "Korrekt",
                "entailment": "Entailment",
                "neutral": "Neutral",
                "contradiction": "Motsägelse",
                "coreferring": "Korefererande",
                "different_sense": "Annan betydelse",
                "same_sense": "Samma betydelse",
            }

            prompt = ""
            sys_prompt = ""
            # Safely get label, handle list or dict if malformed, default string
            answer_raw = row.get("label", "")
            ans_str = str(answer_raw) if answer_raw is not None else ""
            answer = LABEL_MAP.get(ans_str, ans_str)
            options = ["", "", "", "", ""]

            if conf == "absabank-imm":
                sys_prompt = "Bedöm hur positivt eller negativt författaren förhåller sig till invandring på en skala från 1 till 5."
                prompt = f"Text: {row.get('text', '')}"

            elif conf == "argumentation-sentences":
                sys_prompt = (
                    "Bestäm om meningen är för, emot eller orelaterad till ämnet."
                )
                prompt = (
                    f"Ämne: {row.get('topic', '')}\nMening: {row.get('sentence', '')}"
                )

            elif conf == "dalaj-ged-superlim":
                sys_prompt = (
                    "Bestäm om meningen är språkligt korrekt svenska eller inte."
                )
                prompt = f"Mening: {row.get('sentence', '')}"

            elif conf == "sweanalogy":
                sys_prompt = "Givet ett ordpar A:B och ett ord C, hitta ett ord D så att A:B = C:D."
                prompt = f"{row.get('pair1_element1', '')}:{row.get('pair1_element2', '')} = {row.get('pair2_element1', '')}:?"

            elif conf in ["swediagnostics", "swenli", "swewinogender"]:
                sys_prompt = "Bestäm den logiska relationen mellan de två meningarna."
                prompt = f"Premiss: {row.get('premise', '')}\nHypotes: {row.get('hypothesis', '')}"

            elif conf == "swefaq":
                sys_prompt = (
                    "Välj det mest passande svaret på frågan bland alternativen."
                )
                q = row.get("question", "")
                cands = row.get("candidate_answers", [])
                if not isinstance(cands, list):
                    cands = []
                prompt = f"Fråga: {q}\n\nAlternativ:\n"
                letters = ["A", "B", "C", "D", "E"]
                for idx, cand in enumerate(cands[:5]):
                    prompt += f"{letters[idx]}: {cand}\n"
                    options[idx] = str(cand)

                ans_idx = row.get("label")
                if (
                    ans_idx is not None
                    and isinstance(ans_idx, int)
                    and 0 <= ans_idx < len(letters)
                ):
                    answer = letters[ans_idx]

            elif conf == "sweparaphrase":
                sys_prompt = (
                    "Bedöm hur lika de två meningarna är på en kontinuerlig skala."
                )
                prompt = f"Mening 1: {row.get('sentence_1', '')}\nMening 2: {row.get('sentence_2', '')}"

            elif conf == "swesat-synonyms":
                sys_prompt = "Välj rätt synonym till ordet."
                prompt = f"Ord: {row.get('item', '')}\n\nAlternativ:\n"
                cands = row.get("candidate_answers", [])
                if not isinstance(cands, list):
                    cands = []
                letters = ["A", "B", "C", "D", "E"]
                for idx, cand in enumerate(cands[:5]):
                    prompt += f"{letters[idx]}: {cand}\n"
                    options[idx] = str(cand)

                ans_idx = row.get("label")
                if (
                    ans_idx is not None
                    and isinstance(ans_idx, int)
                    and 0 <= ans_idx < len(letters)
                ):
                    answer = letters[ans_idx]

            elif conf == "swewic":
                sys_prompt = "Bestäm om det angivna ordet har samma betydelse i båda kontexterna."
                first = row.get("first", {}) or {}
                second = row.get("second", {}) or {}
                w1 = first.get("word", {}).get("text", "")
                c1 = first.get("context", "")
                w2 = second.get("word", {}).get("text", "")
                c2 = second.get("context", "")
                prompt = f"Ord 1: {w1} i kontext: {c1}\nOrd 2: {w2} i kontext: {c2}"

            elif conf == "swewinograd":
                sys_prompt = "Bestäm om pronomenet syftar på kandidaten i texten."
                prompt = f"Text: {row.get('text', '')}\nPronomen: {row.get('pronoun', {}).get('text', '')}\nKandidat: {row.get('candidate_antecedent', {}).get('text', '')}"
            else:
                # If we don't have a mapping for it, skip.
                continue

            prompt += "\n\nSvar:"

            unified_items.append(
                {
                    "uid": f"superlim2-{conf}-{i}",
                    "test_id": f"superlim2-{conf}",
                    "section": conf,
                    "subsection": conf,
                    "question_id": i,
                    "question_resource": None,
                    "question": prompt,
                    "option_a": options[0],
                    "option_b": options[1],
                    "option_c": options[2],
                    "option_d": options[3],
                    "option_e": options[4],
                    "system_prompt": sys_prompt,
                    "prompt": prompt,
                    "answer": answer,
                    "source": "superlim-2",
                }
            )
            count += 1

        print(f"  -> Processed {count} items for {conf}.")

    print(f"\nSuccessfully mapped {len(unified_items)} total items from superlim-2.")

    output_file = "merged_benchmark.jsonl"
    if os.path.exists(output_file):
        seen = set()
        existing_items = []
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                sig = (item.get("prompt", ""), item.get("answer", ""))
                seen.add(sig)
                existing_items.append(item)

        added = 0
        for item in unified_items:
            sig = (item.get("prompt", ""), item.get("answer", ""))
            if sig not in seen:
                seen.add(sig)
                existing_items.append(item)
                added += 1

        with open(output_file, "w", encoding="utf-8") as f:
            for item in existing_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(
            f"Added {added} unique items. Skipped {len(unified_items) - added} duplicates."
        )
        print(f"Total unique questions in {output_file}: {len(existing_items)}")
    else:
        print(f"{output_file} not found. Could not append to it.")


if __name__ == "__main__":
    map_superlim()
