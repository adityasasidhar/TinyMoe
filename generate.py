import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import ModelArgs, TinyMoE, precompute_rotary_emb


@torch.no_grad()
def generate(model, prompt, freqs_cis, max_new_tokens=50, temperature=1.0, top_k=50):
    model.eval()
    idx = prompt
    seq_len = idx.size(1)

    device_type = "cuda" if next(model.parameters()).is_cuda else "cpu"

    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        # Prefill
        logits, _, past_kvs = model(idx, freqs_cis, start_pos=0, use_cache=True)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx = torch.cat([idx, next_token], dim=1)

        # Decode
        for i in range(max_new_tokens - 1):
            curr_token = idx[:, -1:]
            curr_pos = seq_len + i

            logits, _, past_kvs = model(
                curr_token, freqs_cis, start_pos=curr_pos, use_cache=True, past_kvs=past_kvs
            )

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)

    return idx


def load_model(checkpoint_path: str, device: str, args: ModelArgs) -> TinyMoE:
    model = TinyMoE(args).to(device=device, dtype=torch.bfloat16)

    compiled_state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Strip the '_orig_mod.' prefix added by torch.compile
    clean_state_dict = {}
    for key, value in compiled_state_dict.items():
        clean_key = key.replace("_orig_mod.", "")
        clean_state_dict[clean_key] = value

    model.load_state_dict(clean_state_dict)
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("alea-institute/kl3m-tokenizer-003-16k")

    args = ModelArgs()
    args.vocab_size = len(tokenizer)

    print("Loading model weights (tinymoe_final.pt)...")
    model = load_model("tinymoe_final.pt", device, args)
    model.eval()

    freqs_cis = precompute_rotary_emb(args.dim // args.n_heads, args.seq_len * 2, device)

    print("\n=======================================================")
    print("🧠 TinyMoE is online! Type 'quit' or 'exit' to stop.")
    print("=======================================================\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            if not user_input.strip():
                continue

            prompt_ids = tokenizer.encode(user_input, add_special_tokens=True)
            x = torch.tensor([prompt_ids], dtype=torch.long).to(device)

            output_ids = generate(
                model=model,
                prompt=x,
                freqs_cis=freqs_cis,
                max_new_tokens=60,
                temperature=0.8,
                top_k=40,
            )

            generated_ids = output_ids[0].tolist()
            response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            final_answer = response_text[len(user_input):].strip()

            print(f"\nTinyMoE: {final_answer}\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
