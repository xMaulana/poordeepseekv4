import os
import argparse
import time
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from config import DeepSeekV4Config
from model import DeepSeekV4Model


def load_model(checkpoint_path: str, device: torch.device):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "config" in ckpt:
        config = ckpt["config"]
        print(f"  Config loaded from checkpoint")
    else:
        config = DeepSeekV4Config()
        print(f"  Using default config")
        
    state_dict = ckpt["model_state_dict"]
    has_moe = any("experts" in k for k in state_dict.keys())
    if has_moe and not getattr(config, "use_moe", True):
        print("Checkpoint contains MoE layers but config.use_moe=False. Forcing use_moe=True.")
        config.use_moe = True
    elif not has_moe and getattr(config, "use_moe", True):
        print("Checkpoint lacks MoE layers but config.use_moe=True. Forcing use_moe=False.")
        config.use_moe = False

    model = DeepSeekV4Model(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    step = ckpt.get("step", "?")
    loss = ckpt.get("best_loss", "?")
    print(f"  Checkpoint step: {step} | best loss: {loss}")
    return model, config


def load_untrained_model(device: torch.device):
    """Create a fresh untrained model (for testing generation pipeline)."""
    config = DeepSeekV4Config()
    model = DeepSeekV4Model(config).to(device)
    model.eval()
    print("Using untrained model, output will be random gibberish")
    return model, config


@torch.no_grad()
def generate(
    model: DeepSeekV4Model,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int = 64,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    device: torch.device = torch.device("cpu"),
    use_amp: bool = False,
):
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    if input_ids.shape[1] == 0:
        print("Error: prompt encoded to zero tokens")
        return ""

    max_pos = model.config.max_position_embeddings
    # max_pos = 512
    if input_ids.shape[1] >= max_pos:
        print(f"Warning: prompt ({input_ids.shape[1]} tokens) exceeds max position ({max_pos}), truncating")
        input_ids = input_ids[:, -max_pos + 1:]

    generated_ids = input_ids[0].tolist()
    eos_id = model.config.eos_token_id
    amp_dtype = torch.float16 if use_amp else torch.float32

    print(f"\n{'-' * 70}")
    print(f"Prompt: \"{prompt}\"")
    print(f"Settings: temp={temperature}, top_k={top_k}, top_p={top_p}, rep_pen={repetition_penalty}")
    print(f"Max tokens: {max_tokens} | Max position: {max_pos}")
    print(f"{'-' * 70}\n")
    print(f" {prompt}", end="", flush=True)

    t0 = time.time()
    tokens_generated = 0

    for _ in range(max_tokens):
        if input_ids.shape[1] > max_pos:
            input_ids = input_ids[:, -max_pos:]

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(input_ids)

        logits = outputs["logits"][:, -1, :].float()

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        logits = logits.clamp(-1e4, 1e4)

        if repetition_penalty != 1.0:
            for token_id in set(generated_ids):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        if temperature <= 1e-6:
            next_id = logits.argmax(dim=-1).item()
        else:
            logits = logits / temperature

            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                threshold = top_k_vals[:, -1].unsqueeze(-1)
                logits[logits < threshold] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")

                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            if torch.isnan(probs).any() or probs.sum() < 1e-8:
                next_id = logits.argmax(dim=-1).item()
            else:
                next_id = torch.multinomial(probs, num_samples=1).item()

        if next_id == eos_id:
            break

        generated_ids.append(next_id)
        tokens_generated += 1

        new_text = tokenizer.decode([next_id])
        print(new_text, end="", flush=True)

        next_token = torch.tensor([[next_id]], dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    elapsed = time.time() - t0
    tok_per_sec = tokens_generated / max(elapsed, 1e-6)

    full_text = tokenizer.decode(generated_ids)

    print(f"\n\n{'-'*60}")
    print(f"Generated {tokens_generated} tokens in {elapsed:.2f}s ({tok_per_sec:.1f} tok/s)")

    return full_text



def main():
    parser = argparse.ArgumentParser(
        description="Mini DeepSeek-V4 Text Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate.py --prompt "Indonesia adalah"
  python generate.py --prompt "Pada suatu hari" --temperature 0.7 --top_k 50
  python generate.py --interactive
  python generate.py --checkpoint checkpoints/final_model.pt --prompt "Jakarta"
        """,
    )

    parser.add_argument("--checkpoint", type=str, default="checkpoints/final_model.pt",
                        help="Path to model checkpoint (default: checkpoints/final_model.pt)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")

    parser.add_argument("--prompt", type=str, default=None,
                        help="Input prompt text")
    parser.add_argument("--max_tokens", type=int, default=64,
                        help="Maximum tokens to generate (default: 64)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8, use 0 for greedy)")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-K sampling (default: 50, 0=disabled)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-P nucleus sampling (default: 0.9, 1.0=disabled)")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="Repetition penalty (default: 1.1, 1.0=disabled)")

    parser.add_argument("--tokenizer", type=str, default="tools/tokenizer",
                        help="Path to tokenizer directory")

    args = parser.parse_args()

    if args.cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    print("=" * 60)
    print("  Mini DeepSeek-V4 — Text Generation")
    print("=" * 60)
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")

    tokenizer_file = f"{args.tokenizer}/tokenizer.json"
    tokenizer = Tokenizer.from_file(tokenizer_file)
    print(f"  Tokenizer: vocab_size={tokenizer.get_vocab_size()}")

    if os.path.exists(args.checkpoint):
        model, config = load_model(args.checkpoint, device)
    else:
        print(f"\nCheckpoint not found: {args.checkpoint}")
        model, config = load_untrained_model(device)

    use_amp = device.type == "cuda"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print()

    if args.prompt:
        generate(
            model, tokenizer, args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=device,
            use_amp=use_amp,
        )
    else:
        demo_prompts = [
            "Indonesia adalah",
            "Pada suatu hari",
            "Jakarta merupakan",
        ]
        for prompt in demo_prompts:
            generate(
                model, tokenizer, prompt,
                max_tokens=32,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                device=device,
                use_amp=use_amp,
            )
            print("\n")


if __name__ == "__main__":
    main()
