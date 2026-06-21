import os
import sys
import math
import time
import argparse
import csv
import torch
import torch.nn as nn

from tqdm import tqdm

from config import DeepSeekV4Config
from model import DeepSeekV4Model
from data import create_dataloader
from muon_optimizer import split_params_for_optimizer

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_num(n):
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)

def train_model(config: DeepSeekV4Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\n" + "=" * 60)
    print("Building Mini DeepSeek-V4...")
    print("=" * 60)
    model = DeepSeekV4Model(config).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"\nTotal params: {format_num(total_params)} ({total_params:,})")
    print(f"Trainable:    {format_num(trainable_params)} ({trainable_params:,})")

    print(f"\nSetting up optimizers ({config.optimizer_type})...")
    if config.optimizer_type == "adamw":
        adamw_opt = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=config.weight_decay,
        )
        muon_opt = None
    else:
        optimizers = split_params_for_optimizer(
            model,
            muon_lr=config.muon_lr,
            adamw_lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        muon_opt = optimizers["muon"]
        adamw_opt = optimizers["adamw"]

    print(f"\nLoading dataset: {config.dataset_name}")
    print(f"Tokenizer: {config.tokenizer_path}")
    dataloader = create_dataloader(
        dataset_name=config.dataset_name,
        dataset_text_column=config.dataset_text_column,
        max_rows=config.max_rows,
        tokenizer_path=config.tokenizer_path,
        seq_length=config.seq_length,
        batch_size=config.batch_size,
        load_from_disk=config.load_from_disk
    )

    total_steps_in_epoch = len(dataloader) // config.gradient_accumulation_steps
    if getattr(config, 'max_steps', None) is None or config.max_steps <= 0:
        config.max_steps = max(1, total_steps_in_epoch)
        print(f"Auto-calculated max_steps: {config.max_steps} (1 epoch)")

    adamw_sched = get_cosine_schedule_with_warmup(adamw_opt, config.warmup_steps, config.max_steps)
    if muon_opt is not None:
        muon_sched = get_cosine_schedule_with_warmup(muon_opt, config.warmup_steps, config.max_steps)
    else:
        muon_sched = None

    use_amp = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16
    print(f"Mixed precision: {use_amp} (dtype={amp_dtype})")

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    csv_path = os.path.join(config.log_dir, "history.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "train_loss", "eval_loss", "perplexity", "tok_per_sec", "mode"])

    open(os.path.join(config.log_dir, "train.log"), "w").close()

    print("\n" + "=" * 60)
    print(f"Training config:")
    print(f"  Seq length:    {config.seq_length}")
    print(f"  Batch size:    {config.batch_size}")
    print(f"  Grad accum:    {config.gradient_accumulation_steps}")
    print(f"  Effective BS:  {config.batch_size * config.gradient_accumulation_steps}")
    print(f"  Max steps:     {config.max_steps}")
    print(f"  Warmup steps:  {config.warmup_steps}")
    print(f"  Muon LR:       {config.muon_lr}")
    print(f"  AdamW LR:      {config.learning_rate}")
    print("=" * 60 + "\n")

    model.train()
    global_step = 0
    accum_loss = 0.0
    best_loss = float("inf")
    start_time = time.time()
    tokens_processed = 0

    data_iter = iter(dataloader)
    pbar = tqdm(total=config.max_steps, desc="Training", unit="step")

    while global_step < config.max_steps:
        if muon_opt is not None:
            muon_opt.zero_grad()
        adamw_opt.zero_grad()

        for micro_step in range(config.gradient_accumulation_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
            else:
                input_ids, labels = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids, labels=labels, use_checkpoint=True)
                loss = outputs["loss"] / config.gradient_accumulation_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item()
            tokens_processed += input_ids.numel()

        if muon_opt is not None:
            scaler.unscale_(muon_opt)
        scaler.unscale_(adamw_opt)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print(f"\n[WARNING] NaN/Inf grad at step {global_step+1}, skipping update")
            if muon_opt is not None:
                muon_opt.zero_grad()
            adamw_opt.zero_grad()
            scaler.update()
        else:
            if muon_opt is not None:
                scaler.step(muon_opt)
            scaler.step(adamw_opt)
            scaler.update()

        if muon_sched is not None:
            muon_sched.step()
        adamw_sched.step()

        global_step += 1
        pbar.update(1)

        if global_step % config.log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_processed / elapsed
            current_adamw_lr = adamw_sched.get_last_lr()[0]
            current_muon_lr = muon_sched.get_last_lr()[0] if muon_sched is not None else 0.0

            avg_loss = accum_loss / config.log_interval

            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "grad": f"{grad_norm:.2f}",
                "tok/s": f"{tokens_per_sec:.0f}",
                "lr": f"{current_adamw_lr:.2e}",
            })

            log_line = (
                f"step={global_step} | loss={avg_loss:.4f} | "
                f"grad_norm={grad_norm:.4f} | tok/s={tokens_per_sec:.0f} | "
                f"muon_lr={current_muon_lr:.6f} | adamw_lr={current_adamw_lr:.6f}\n"
            )
            with open(os.path.join(config.log_dir, "train.log"), "a") as f:
                f.write(log_line)

            csv_writer.writerow([global_step, f"{avg_loss:.4f}", "", "", f"{tokens_per_sec:.0f}", "vanilla_train"])
            csv_file.flush()

            if avg_loss < best_loss:
                best_loss = avg_loss

            accum_loss = 0.0

        if global_step % config.save_interval == 0:
            ckpt_path = os.path.join(config.output_dir, f"step_{global_step}.pt")
            torch.save({
                "step": global_step,
                "model_state_dict": model.state_dict(),
                "muon_state_dict": muon_opt.state_dict() if muon_opt is not None else None,
                "adamw_state_dict": adamw_opt.state_dict(),
                "config": config,
                "best_loss": best_loss,
            }, ckpt_path)
            print(f"\nSaved checkpoint: {ckpt_path}")

    pbar.close()
    csv_file.close()
    elapsed = time.time() - start_time

    final_path = os.path.join(config.output_dir, "final_model.pt")
    torch.save({
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "config": config,
        "best_loss": best_loss,
    }, final_path)

    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"  Total steps:  {global_step}")
    print(f"  Best loss:    {best_loss:.4f}")
    print(f"  Total time:   {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"  Total tokens: {format_num(tokens_processed)}")
    print(f"  Saved to:     {final_path}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Mini DeepSeek-V4 Training")

    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--num_hidden_layers", type=int, default=None)
    parser.add_argument("--num_attention_heads", type=int, default=None)
    parser.add_argument("--num_key_value_heads", type=int, default=None)
    parser.add_argument("--use_moe", type=lambda x: (str(x).lower() == 'true'), default=None, help="Set to False to disable MoE completely")
    parser.add_argument("--moe_layer_freq", type=int, default=None, help="Set to 999 to disable MoE layers")
    parser.add_argument("--n_routed_experts", type=int, default=None)
    parser.add_argument("--n_shared_experts", type=int, default=None)
    parser.add_argument("--num_experts_per_tok", type=int, default=None)
    parser.add_argument("--moe_intermediate_size", type=int, default=None)
    parser.add_argument("--intermediate_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seq_length", type=int, default=None)
    parser.add_argument("--max_position_embeddings", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--muon_lr", type=float, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--optimizer_type", type=str, default=None,
                        choices=["muon", "adamw"],
                        help="Optimizer: 'muon' or 'adamw' (PyTorch, no CUDA compile)")

    args = parser.parse_args()
    config = DeepSeekV4Config()

    for key, val in vars(args).items():
        if key == "local_rank":
            continue
        if val is not None and hasattr(config, key):
            setattr(config, key, val)
            print(f"Override: {key} = {val}")

    if args.hidden_size or args.num_attention_heads:
        config.head_dim = config.hidden_size // config.num_attention_heads

    if config.seq_length > config.max_position_embeddings:
        config.max_position_embeddings = config.seq_length
        print(f"Auto-adjusted max_position_embeddings to {config.max_position_embeddings} to match seq_length")

    train_model(config)


if __name__ == "__main__":
    main()
