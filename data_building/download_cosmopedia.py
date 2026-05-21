from datasets import load_dataset
import json
from tqdm import tqdm

TARGET_SIZE_MB = 

OUTPUT_FILE = "cosmopedia_300mb.jsonl"

target_bytes = TARGET_SIZE_MB * 1024 * 1024

written_bytes = 0

dataset = load_dataset(
    "HuggingFaceTB/cosmopedia",
    "web_samples_v2",
    split="train",
    streaming=True
)

with open(OUTPUT_FILE, "w") as f:

    for sample in tqdm(dataset):

        text = sample["text"].strip()

        if len(text) < 50:
            continue

        item = {
            "text": text
        }

        line = json.dumps(item) + "\n"

        encoded = line.encode("utf-8")

        f.write(line)

        written_bytes += len(encoded)

        if written_bytes >= target_bytes:
            break

print(
    f"Saved "
    f"{written_bytes / 1024 / 1024:.2f} MB"
)