# Running DeepSeek-V4 on a Potato (Well, a Laptop)

The industry is currently obsessed with stuffing 1.6 trillion parameters into warehouse-sized data centers, and frankly? It's exhausting to keep up with. 

I didn't want to spin up a massive cluster just to figure out what the folks at DeepSeek were actually doing under the hood. So basically, I stripped their architecture down to the absolute studs. This is a miniaturized, from-scratch implementation of the DeepSeek-V4 model. It clocks in at roughly 70.9M parameters (or around 61.5M if you rip out the MoE layers, which I sometimes do when I'm feeling particularly lazy about dense training dynamics).

I built this specifically to run on my daily driver, a rather mediocre laptop packing an NVIDIA RTX 3050 with a measly 4GB of VRAM. It fits. Barely, but it fits.

The point isn't to build a chatbot that's going to steal your job. Obviously. The point is to get our hands dirty and actually see how the math works when you aren't hiding behind an API abstraction layer.

---

## The Stats: V4 vs My Hack Job

Here's the rundown of how much I had to butcher the original specs to make this run without melting my desk.

| Component | The "Real" V4 | My Compromise |
| :--- | :--- | :--- |
| **Total Parameters** | 1.6T (49B active) | ~70.9M with MoE (~61.5M when I turn it off) |
| **Hidden Size** | 7,168 | 512 |
| **Transformer Layers** | 61 | 8 |
| **Attention Heads** | 128 Q / 1 KV | 8 Q / 4 KV (standard GQA stuff) |
| **Head Dimension** | 512 | 64 |
| **Routed Experts** | 384 | 4 (just enough to see it route) |
| **Shared Experts** | 1 | 1 |
| **Experts per Token** | 6 | 2 |
| **Expert FFN Size** | 3,072 | 512 |
| **Dense FFN Size** | x | 1,024 |
| **Vocab Size** | 129,280 | 16,384 (I trained a custom BPE on some Indonesian poetry I had lying around) |
| **Context Length** | 1,000,000 tokens | 1024 tokens. Let's be realistic here. |
| **Sliding Window** | 128 | 64 |
| **mHC Multiplier** | 4 | 4 |
| **Sinkhorn Iterations** | 20 | 20 |
| **Hash Routing Layers** | First 3 MoE layers | Layers 1 and 3 |
| **Scoring Function** | `sqrt(softplus)` | `sqrt(softplus)` |
| **Optimizer** | Muon + AdamW | Configurable (I use Muon+AdamW, but you can stick to pure AdamW if you're stubborn) |
| **Precision** | BF16 / FP4 | BF16 / FP16 Mixed Precision (AMP) |

---

## What's Actually Under the Hood?

If you dig into the code, you'll find the core innovations. I didn't skip the hard parts, even if it took me three cups of coffee to get the gradients to behave.

### 1. Manifold-Constrained Hyper-Connections (mHC)
They ditched the old standard residual connections (`x = x + sublayer(x)`). Now we mix streams using a learned matrix that's projected into the doubly stochastic space via the Sinkhorn-Knopp algorithm. Sounds fancy, but it just keeps the spectral norm bounded so the gradients don't blow up.

You can poke around in [modules.py](file:///home/xmaulana/dataku/ngoding/ngonten/deepseekv4poor/modules.py#L51-L84).
```python
# 1. Expand the hidden state 
x_expanded = self.proj_in(x)  
x_streams = x_expanded.view(batch, seq_len, self.hc_mult, self.hidden_size)

# 2. Shove the sublayer output into the first stream
stream_0 = x_streams[:, :, 0:1, :] + sublayer_output.unsqueeze(2)
x_streams = torch.cat([stream_0, x_streams[:, :, 1:, :]], dim=2)

# 3. Sinkhorn-Knopp mixing (I force this to run in float32 so we don't underflow and die)
mixing = self.sinkhorn_knopp(self.log_alpha)
x_mixed = torch.einsum("ij,bsjd->bsid", mixing, x_streams)

# 4. Project back
x_flat = x_mixed.reshape(batch, seq_len, self.expanded_size)
return x + sublayer_output + self.proj_out(x_flat)
```
Quick aside, if you don't run that Sinkhorn loop in float32, fp16 will absolutely wreck your day with underflow errors. Ask me how I know.

### 2. DeepSeekMoE 
I alternate between dense layers and MoE layers because, frankly, computing everything everywhere is a waste of cycles. 
Instead of sigmoid, they use `sqrt(softplus(x))` for the routing scores. It supposedly gives better gradient flow. Seems to work fine. Also, I hardcoded deterministic hash routing for the first two MoE layers just to force the experts to actually balance out early on instead of collapsing into a lazy consensus. 

### 3. Hybrid Attention (CSA + SWA)
The memory footprint of a full KV cache is stupid. To fix it, the attention alternates. Even layers compress the KV states by a factor of 4 (CSA). Odd layers compress it by 16 (HCA). I also threw in a 64-token sliding window attention branch that runs parallel to catch the local details. It's a bit of a Frankenstein setup, but it drastically cuts memory usage.

### 4. Splitting the Optimizers
Standard AdamW wasn't enough for them, so they brought in Muon. It uses Newton-Schulz iterations to orthogonalize the 2D weight updates. I split the parameters, Muon handles the 2D weights (attention, MLPs, routing gates), and standard AdamW handles the 1D stuff and embeddings. 

If you hate new things, you can disable Muon with a CLI flag. The splitting logic is in [muon_optimizer.py](file:///home/xmaulana/dataku/ngoding/ngonten/deepseekv4poor/muon_optimizer.py) if you want to judge my implementation.

---

## The Files (Because you're going to ask)

I kept the directory structure flat. I hate nesting files for no reason.

- [config.py](file:///home/xmaulana/dataku/ngoding/ngonten/deepseekv4poor/config.py): The hyperparams. Defaults to training on an Indonesian poetry CSV I found. 
- [modules.py](file:///home/xmaulana/dataku/ngoding/ngonten/deepseekv4poor/modules.py): The actual guts of the network (Norms, RoPE, the MoE stuff).
- [model.py](file:///home/xmaulana/dataku/ngoding/ngonten/deepseekv4poor/model.py): Stitches the modules together.
- [muon_optimizer.py](file:///home/xmaulana/dataku/ngoding/ngonten/deepseekv4poor/muon_optimizer.py): My parameter splitting duct-tape.
- [data.py](file:///home/xmaulana/dataku/ngoding/ngonten/deepseekv4poor/data.py): Just a barebones dataloader that chunks text on the fly.
- [train_tokenizer.py](file:///home/xmaulana/dataku/ngoding/ngonten/deepseekv4poor/train_tokenizer.py): Builds the 16k vocab.
- [train.py](file:///home/xmaulana/dataku/ngoding/ngonten/deepseekv4poor/train.py): The main loop. Handles the AMP scaling and spits out logs.
- [generate.py](file:///home/xmaulana/dataku/ngoding/ngonten/deepseekv4poor/generate.py): A rudimentary script to see if the model learned anything besides static noise.
- [test.py](file:///home/xmaulana/dataku/ngoding/ngonten/deepseekv4poor/test.py): A tiny smoke test just to make sure the backward pass doesn't immediately NaN out on your GPU.

---

## How to actually run this thing

I use `uv` because `pip` is painfully slow and life is short. 

1. **Install the dependencies:**
   Just run `uv sync`. It'll handle the virtual env. If you refuse to use uv, well... you know how to read a `pyproject.toml` file.

2. **Smoke test it:**
   Run `uv run python test.py`. If it errors out, your CUDA setup is probably messed up.

3. **Train the tokenizer:**
   `uv run python train_tokenizer.py`
   This is mandatory. By default it chews through the `data/puisi.csv` poetry file.

4. **Train the actual model:**
   `uv run python train.py`
   That's it. If you want to disable the MoE stuff because you just want a dense baseline, run:
   `uv run python train.py --use_moe False --optimizer_type adamw --max_steps 1000`

5. **Generate something:**
   `uv run python generate.py --checkpoint checkpoints/final_model.pt --prompt "Pada suatu hari"`

---

## The FP16 Nightmare

Look, doing custom routing and doubly stochastic matrices in mixed precision on consumer hardware is an absolute nightmare. I spent way too many hours chasing NaN explosions. Here's a cheat sheet of the landmines I stepped on and how I jerry-rigged a fix:

| The Problem | Why it blew up in my face | The Duct Tape |
| :--- | :--- | :--- |
| `sqrt(softplus(x))` | `sqrt(0)` gives an infinite derivative. Instant NaN gradients. | Shoved a tiny epsilon in there: `torch.sqrt(F.softplus(x) + 1e-6)` |
| Sinkhorn iterations | Exponentials inside a logsumexp underflow almost immediately in fp16. | Forced the whole loop into `.float()` and cast it back at the end. |
| MoE top-k norm | Tiny routing scores round to exactly 0.0 in fp16. Division by zero ensues. | Beefed up the stabilizer: `+ 1e-5` in the denominator. |
| Checkpointing failures | Modifying the mHC streams in-place broke PyTorch's gradient checkpointing. | Swapped the slice mutation for a somewhat clunkier `torch.cat`. It works. |
| Exploding variance | Standard init makes the deep projections scale to infinity. | Squashed the standard deviation of `o_proj`, `down_proj`, and `proj_out` by `1/sqrt(2 * num_layers)`. |
| Random gradient spikes | Sometimes a batch is just cursed and poisons the optimizer state forever. | Slapped a NaN/Inf detector in `train.py`. If it triggers, it just throws the step in the trash and moves on. |

---

## Managing Expectations

I mean this sincerely: temper your expectations.

You're training a 70M parameter model on a laptop graphics card using a tiny text dataset. 
The initial loss is going to be sitting around 11.8 (which makes sense, given the 16k vocab). If you let it run for a while, you might drag it down to the 5.0 range. 

Is the output going to rival GPT-4? Not even close. It'll probably stutter out repetitive garbage most of the time. But if you watch closely, you'll start seeing it string together actual Indonesian grammar patterns. And honestly? Seeing a miniature version of a trillion-parameter architecture slowly figure out punctuation on a $300 GPU is pretty damn satisfying.

## Reading Material

If you want to read the real math instead of my tired explanations:
* The DeepSeek-V4 Technical Report
* "Old Optimizer, New Norm" (Bernstein & Newhouse, 2024) - for the Muon stuff.
* Dig into Sinkhorn-Knopp if you're bored and like linear algebra.
