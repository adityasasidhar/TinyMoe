import os

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ModelArgs, TinyMoE, precompute_rotary_emb
from generate import generate
from data import CosmopediaDataset


# ---- Config ----

TOKENIZER      = "alea-institute/kl3m-tokenizer-003-16k"
DATASET_PATH   = "cosmopedia_1gb.jsonl"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE     = 8
GRAD_ACCUM     = 8
LR             = 2e-4
WEIGHT_DECAY   = 0.01
EPOCHS         = 3
SAVE_EVERY     = 3000
CHECKPOINT_DIR = "checkpoints"


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ---- Tokenizer ----
    print(f"Loading tokenizer {TOKENIZER}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ---- Model ----
    args = ModelArgs(vocab_size=len(tokenizer))
    print(f"Tokenizer vocab size locked at: {args.vocab_size}")

    model = TinyMoE(args).to(device=DEVICE, dtype=torch.bfloat16)
    freqs_cis = precompute_rotary_emb(args.dim // args.n_heads, args.seq_len, DEVICE)

    if DEVICE == "cuda":
        print("Compiling model via Inductor Engine...")
        model = torch.compile(model, mode="max-autotune-no-cudagraphs")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params / 1e6:.2f}M")

    # ---- Data ----
    dataset = CosmopediaDataset(DATASET_PATH, tokenizer, args.seq_len)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
    )

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = (len(loader) // GRAD_ACCUM) * EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # ---- Training Loop ----
    global_step = 0
    model.train()

    for epoch in range(EPOCHS):
        pbar = tqdm(loader)
        for step, (x, y) in enumerate(pbar):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
                _, loss, _ = model(x, freqs_cis, targets=y)
                loss = loss / GRAD_ACCUM

            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            pbar.set_description(f"epoch={epoch} loss={loss.item() * GRAD_ACCUM:.4f}")

            if global_step > 0 and global_step % SAVE_EVERY == 0:
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": global_step,
                }
                torch.save(checkpoint, f"{CHECKPOINT_DIR}/tinymoe_step_{global_step}.pt")

    torch.save(model.state_dict(), "tinymoe_final.pt")
    print("\nTraining complete")

    # ---- Generation Test ----
    print("Testing Generation...")
    model.eval()
    prompt = "Explain gravity simply."

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    x = torch.tensor([prompt_ids], dtype=torch.long).to(DEVICE)

    gen_freqs_cis = precompute_rotary_emb(args.dim // args.n_heads, args.seq_len * 2, DEVICE)
    output_ids = generate(model, x, gen_freqs_cis, max_new_tokens=100)

    generated_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

    print("\n====================")
    print(generated_text)
    print("====================")


if __name__ == "__main__":
    main()