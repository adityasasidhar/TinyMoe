"""
Download a diverse mixture of Cosmopedia splits for MoE training.

The idea: different domains have very different token distributions, which
encourages individual experts to specialize (e.g., one expert learns math
notation, another learns narrative structure, another learns procedural
how-to steps). A homogeneous dataset would give the router no reason to
differentiate.

Mixture (1 GB total):
    web_samples_v2  250 MB  — general web knowledge (backbone)
    auto_math_text  200 MB  — math / reasoning (distinct symbolic domain)
    stories         150 MB  — creative / narrative writing
    openstax        150 MB  — structured textbook content
    wikihow         125 MB  — procedural how-to guides
    stanford        125 MB  — academic / lecture-style
"""

import json
import random

from datasets import load_dataset
from tqdm import tqdm

# (config_name, target_bytes)
MIXTURE = [
    ("web_samples_v2",  250),
    ("auto_math_text",  50),
    ("stories",         200),
    ("openstax",        150),
    ("wikihow",         175),
    ("stanford",        175),
]

OUTPUT_FILE = "cosmopedia_1gb.jsonl"
MB = 1024 * 1024


def download_split(config: str, target_mb: int):
    """Stream one Cosmopedia split and collect samples up to target size."""
    target_bytes = target_mb * MB
    written = 0
    samples = []

    dataset = load_dataset(
        "HuggingFaceTB/cosmopedia",
        config,
        split="train",
        streaming=True,
    )

    print(f"  ↳ {config:20s}  target={target_mb} MB")
    for sample in dataset:
        text = sample["text"].strip()
        if len(text) < 50:
            continue

        item = {"text": text, "source": config}
        line = json.dumps(item) + "\n"
        encoded = line.encode("utf-8")

        samples.append(line)
        written += len(encoded)

        if written >= target_bytes:
            break

    print(f"       collected {written / MB:.1f} MB  ({len(samples)} samples)")
    return samples


def main():
    all_samples = []

    print(f"Downloading {len(MIXTURE)} splits → {OUTPUT_FILE}\n")
    for config, target_mb in MIXTURE:
        samples = download_split(config, target_mb)
        all_samples.extend(samples)

    # Shuffle so domains are interleaved (prevents the model from seeing
    # all math, then all stories, etc. within a single epoch)
    print(f"\nShuffling {len(all_samples)} total samples...")
    random.shuffle(all_samples)

    with open(OUTPUT_FILE, "w") as f:
        for line in all_samples:
            f.write(line)

    total_bytes = sum(len(l.encode("utf-8")) for l in all_samples)
    print(f"Saved {total_bytes / MB:.1f} MB → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()