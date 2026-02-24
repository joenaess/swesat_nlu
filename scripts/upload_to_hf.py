import os
import argparse
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Upload the merged benchmark to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Your Hugging Face repository ID (e.g., 'username/swesat-skolprov-merged')",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (optional if logged in via CLI)",
    )
    args = parser.parse_args()

    print(f"Loading local dataset from 'merged_benchmark.jsonl'...")
    dataset = load_dataset("json", data_files="merged_benchmark.jsonl", split="train")

    print(f"Uploading dataset to {args.repo_id} on Hugging Face Hub...")
    if args.token:
        # If token is provided, set it in environment or pass it
        os.environ["HF_TOKEN"] = args.token

    dataset.push_to_hub(args.repo_id)
    print("Upload complete!")


if __name__ == "__main__":
    main()
