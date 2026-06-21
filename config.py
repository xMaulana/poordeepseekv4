from dataclasses import dataclass
from typing import List


@dataclass
class DeepSeekV4Config:
    vocab_size: int = 16384
    bos_token_id: int = 0
    eos_token_id: int = 1
    pad_token_id: int = 1

    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8 
    num_key_value_heads: int = 4
    head_dim: int = 64
    intermediate_size: int = 1024

    max_position_embeddings: int = 256 
    rope_theta: float = 10000.0
    qk_rope_head_dim: int = 64

    use_moe: bool = True
    moe_layer_freq: int = 2
    n_routed_experts: int = 4
    n_shared_experts: int = 1
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 512
    routed_scaling_factor: float = 2.5
    scoring_func: str = "sqrtsoftplus"
    norm_topk_prob: bool = True

    num_hash_layers: int = 2

    sliding_window: int = 64
    compress_ratio_csa: int = 4
    compress_ratio_hca: int = 16

    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20 
    hc_eps: float = 1e-6

    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    initializer_range: float = 0.02

    dropout: float = 0.0
    tie_word_embeddings: bool = True

    learning_rate: float = 1e-3
    muon_lr: float = 1e-3
    weight_decay: float = 0.01
    max_steps: int = -1
    warmup_steps: int = 200
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 100
    use_amp: bool = True
    amp_dtype: str = "bfloat16"

    optimizer_type: str = "muon"

    load_from_disk: bool = True
    # dataset_name: str = "indonesian-nlp/wikipedia-id"
    # dataset_text_column: str = "text"
    dataset_name: str = "data/puisi.csv"
    dataset_text_column: str = "puisi"
    max_rows: int = 0  
    tokenizer_path: str = "tools/tokenizer"
    seq_length: int = 1024

    output_dir: str = "checkpoints"
    log_dir: str = "logs"

    @property
    def moe_layers(self) -> List[int]:
        if not self.use_moe:
            return []
        return [i for i in range(self.num_hidden_layers) if i % self.moe_layer_freq == 1]

    @property
    def hash_routing_layers(self) -> List[int]:
        moe = self.moe_layers
        return moe[:self.num_hash_layers]

    def get_compress_ratio(self, layer_idx: int) -> int:
        if layer_idx % 2 == 0:
            return self.compress_ratio_csa
        else:
            return self.compress_ratio_hca
