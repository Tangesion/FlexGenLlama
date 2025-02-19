import argparse
import dataclasses
import numpy as np
@dataclasses.dataclass(frozen=True)
class LlamaConfig:
    name: str = "Llama-2-7b-hf"
    num_hidden_layers: int = 32
    max_seq_len: int = 4096
    hidden_size: int = 4096
    n_head: int = 32
    num_key_value_heads: int = 32
    input_dim: int = 4096
    ffn_embed_dim: int = 11008
    pad: int = 1
    activation_fn: str = 'silu'
    vocab_size: int = 32000
    layer_norm_eps: float = 1e-5
    pad_token_id: int = 1
    dtype: type = np.float16

    def model_bytes(self):
        h = self.input_dim
        return 2 * (self.num_hidden_layers * (
        # self-attention
        h * (3 * h + 1) + h * (h + 1) +
        # mlp
        h * (4 * h + 1) + h * 4 * (h + 1) +
        # layer norm
        h * 4) +
        # embedding
        self.vocab_size * (h + 1))

    def cache_bytes(self, batch_size, seq_len):
        return 2 * batch_size * seq_len * self.num_hidden_layers * self.input_dim * 2

    def hidden_bytes(self, batch_size, seq_len):
        return batch_size * seq_len * self.input_dim * 2