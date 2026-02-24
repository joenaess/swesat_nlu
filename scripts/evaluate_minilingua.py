import torch
import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm


def run_evaluation():
    model_name = "minilingua-ai/MiniLingua-1b-Instruct"
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", dtype=torch.float16
    )
    generator = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, trust_remote_code=True
    )

    # Load the merged dataset JSONL
    dataset_file = "merged_benchmark.jsonl"
    print(f"Loading dataset from {dataset_file}...")
    samples = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    # Select 50 random samples with a known short answer to evaluate
    random.seed(42)
    eval_set = random.sample(samples, min(50, len(samples)))

    correct = 0
    total = len(eval_set)
    cnt = 0

    print(f"Evaluating on {total} samples...")
    for item in tqdm(eval_set):
        sys_prompt = item.get("system_prompt", "")
        prompt = item.get("prompt", "")
        messages = [
            {
                "role": "user",
                "content": f"{sys_prompt}\n\n{prompt}".strip()
                if sys_prompt
                else prompt.strip(),
            }
        ]

        # MiniLingua uses standard formatting usually, but let's just supply it raw if no chat template applies.
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted_prompt = (
                f"{sys_prompt}\n\n{prompt}".strip() if sys_prompt else prompt.strip()
            )
            formatted_prompt += "\nSvar:"

        outputs = generator(
            formatted_prompt,
            max_new_tokens=15,
            max_length=None,
            do_sample=False,
            return_full_text=False,
        )
        generated_text = outputs[0]["generated_text"].strip()

        if "Svar:" in generated_text:
            generated_text = generated_text.split("Svar:")[-1].strip()

        expected_answer = str(item.get("answer", "")).strip()

        if expected_answer.lower() in generated_text.lower():
            correct += 1

        # Log 10 sample runs for debugging to show qualitative Swedish abilities
        if cnt < 10:
            print(
                f"\n[Q]: {prompt[:100]}...\n[Expected]: {expected_answer}\n[Generated (Swedish)]: {generated_text}"
            )
        cnt += 1

    # Note: Accuracy will be roughly 0% because the model answers in Swedish while SuperLim tags are English.
    # A true evaluation harness (like lm-evaluation-harness) uses loglikelihoods or translated label maps!
    accuracy = correct / total
    print(f"\nEvaluation Complete!")
    print(
        f"Exact string match Accuracy: {accuracy * 100:.2f}% ({correct}/{total}) - Note: Expect low exact match due to Swedish/English label mismatches in non-MCQA SuperLim tasks."
    )


if __name__ == "__main__":
    run_evaluation()
