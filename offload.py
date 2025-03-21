import torch
import os
import numpy as np
import dataclasses
from typing import Any, Union, Optional, List
from pytorch_backend import (TorchDevice,TorchTensor, general_copy)
from flexgen.utils import ValueHolder, Task, array_1d, array_2d, array_3d, str2bool ,GB, project_decode_latency, write_benchmark_log
from flexgen.opt_config import OptConfig, get_opt_config, download_opt_weights
from flexgen.timer import timers
from transformers import AutoTokenizer
from llama_config import LlamaConfig
##  smart copy (dest, indic) dest:为设备类型，indic:为拷贝的索引  return:返回拷贝后的引用 TorchTensor
# value.smart_copy(des,indic) 为将value拷贝到dest设备上，拷贝的索引为indic 

# python offload.py --model OPT-1.3B --path ../OPT-1.3B-flex --old-path ../OPT-1.3B
# python offload.py --model Llama-2-7b-hf --path ../Llama-2-7b-hf-offload --old-path ../Llama-2-7b-hf

DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes

@dataclasses.dataclass(frozen=True)
class ExecutionEnv:
    """Hardware environment."""
    gpu: TorchDevice = None
    cpu: TorchDevice = None

    @classmethod
    def create(cls, offload_dir):
        # fix recursive import
        
        gpu = TorchDevice("cuda:0")
        cpu = TorchDevice("cpu")
        return cls(gpu=gpu, cpu=cpu)


@dataclasses.dataclass(frozen=True)
class Policy:
    gpu_batch_size: int
    num_gpu_batches: int

    # percent = a means a%
    w_gpu_percent: float
    w_cpu_percent: float
    cache_gpu_percent: float
    cache_cpu_percent: float
    act_gpu_percent: float
    act_cpu_percent: float

    # Whether to overlap the I/O and compute
    overlap: bool

    # Whether to separate attention and mlp as two layers
    sep_layer: bool

    # Whether to use pinned memory for weights on CPU
    pin_weight: bool

    # Whether to compute attention on CPU
    cpu_cache_compute: bool

    # Sparsity of attention weights
    attn_sparsity: float

    # Compress weights with group-wise quantization
    compress_weight: bool
    #comp_weight_config: CompressionConfig

    # Compress KV cache with group-wise quantization
    compress_cache: bool
    #comp_cache_config: CompressionConfig

    @property
    def w_disk_percent(self):
        return 100 - self.w_gpu_percent - self.w_cpu_percent

    @property
    def cache_disk_percent(self):
        return 100 - self.cache_gpu_percent - self.cache_cpu_percent

    @property
    def act_disk_percent(self):
        return 100 - self.act_gpu_percent - self.act_cpu_percent





def init_weight_list(weight_specs: list, policy: Policy, env: ExecutionEnv):
    pin_memory = policy.pin_weight
    home = env.gpu
    ret = []
    for i in range(len(weight_specs)):
        shape, dtype, _ = weight_specs[i]
        weight = home.allocate(shape, dtype, pin_memory=pin_memory)
        weight.load_from_np_file(weight_specs[i][2])
        ret.append(weight)
    return ret

class InputEmbed:
    def __init__(self, config: LlamaConfig, env: ExecutionEnv, policy: Policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home: ValueHolder, path):
        v, h, s, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.max_seq_len, self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_token
            ((v, h), dtype, path + "embed_tokens.weight"),
            # w_pos
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_token = weight_home.val[0]
        w_token: TorchTensor
        if k == 0:
            dst = self.weight_load_dst
            weight_read_buf.store(w_token.smart_copy(dst))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len), np.int64

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        # Compute input embedding
        donate = [False] * 4
        h, donate[0] = hidden.val, True
        mask, donate[1] = attention_mask.val.smart_copy(self.compute)

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_token, donate[2]) = weight_read_buf.pop()
        else:
            (w_token, _) = weight_read_buf.val

        h = self.compute.llama_input_embed(h, mask,
            w_token, self.config.pad_token_id, donate)
        hidden.val = h


class OutputEmbed:
    def __init__(self, config: LlamaConfig, env: ExecutionEnv, policy: Policy):
        self.config = config
        self.env = env
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst = (self.compute.compressed_device if policy.compress_weight
            else self.compute)

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        v, h, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            # w_ln
            ((h,), dtype, path + "norm.weight"),
            # w_token
            ((v, h), dtype, path + "lm_head.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)

        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        w_ln, w_token = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((w_ln.smart_copy(dst2),
                w_token.smart_copy(dst1)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        donate = [False] * 3
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            (w_ln, donate[1]), (w_token, donate[2]) = weight_read_buf.pop()
        else:
            (w_ln, _), (w_token, _) = weight_read_buf.val

        h = self.compute.llama_output_embed(h, w_ln, w_token, donate,
            self.task.do_sample, self.task.temperature)
        hidden.val = h




class SelfAttention:
    def __init__(self, config: LlamaConfig, env: ExecutionEnv, policy: Policy, layer_id: int):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst =  self.compute
        self.attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)
        self.task = None

    def set_task(self, task: Task):
        self.task = task

    def init_weight(self, weight_home: ValueHolder, path: str):
        h, n_head, dtype = (self.config.input_dim, self.config.n_head, self.config.dtype)
        path = os.path.join(path, "")
        #path = os.path.join(os.path.join(path, f"layers.{self.layer_id}.self_attn"))
        #path_lm = os.path.join(os.path.join())
        lm_path = path + f"layers.{self.layer_id}"
        path = path + f"layers.{self.layer_id}.self_attn"
        weight_specs = [
            # w_q
            ((h, h), dtype, path + ".q_proj.weight"),
            # w_k
            ((h, h), dtype, path + ".k_proj.weight"),
            # w_v
            ((h, h), dtype, path + ".v_proj.weight"),
            # w_out
            ((h, h), dtype, path + ".o_proj.weight"),
            # w_ln
            ((h,), dtype, lm_path + ".input_layernorm.weight"),
            # emb_inv_freq
            ((int(h // n_head // 2),), dtype, path + ".rotary_emb.inv_freq"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home: ValueHolder, weight_read_buf: ValueHolder, k):
        w_q: TorchTensor
        w_k: TorchTensor
        w_v: TorchTensor
        w_out: TorchTensor
        w_ln: TorchTensor
        inv_freq: TorchTensor
        w_q, w_k, w_v, w_out, w_ln, inv_freq = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                w_q.smart_copy(dst1),
                w_k.smart_copy(dst1), 
                w_v.smart_copy(dst1), 
                w_out.smart_copy(dst1),
                w_ln.smart_copy(dst1), inv_freq.smart_copy(dst2))) 
            
    def init_cache_one_gpu_batch(self, cache_home: ValueHolder):
        device = self.env.gpu
        cache = device.init_cache_one_gpu_batch(self.config, self.task, self.policy)
        cache_home.store(cache)

    def load_cache(self, cache_home: ValueHolder, cache_read_buf: ValueHolder, i: int):
        if i == 0:  # prefill, no cache
            return
        k_home : TorchTensor
        v_home : TorchTensor

        k_home, v_home = cache_home.val

        # Pick code path
        # path=0 gpu/cpu -> gpu
        # !!!path=1 gpu/cpu -> cpu  
        if self.policy.cpu_cache_compute:
            path = 1
        else:
            path = 0
        dst = self.attention_compute

        if path == 0:  # Direct copy
            # shape: (s, b * n_head, head_dim)
            indices = (slice(0, self.task.prompt_len + i),
                       slice(0, k_home.shape[1]))
            #如果attention矩阵是完全稠密的，则需要将k和v都拷贝到gpu上
            if self.policy.attn_sparsity >= 1.0:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    v_home.smart_copy(dst, indices),
                ))
            #否则，则不需要将v拷贝到gpu上
            else:
                cache_read_buf.store((
                    k_home.smart_copy(dst, indices),
                    (v_home, False),
                ))
        elif path == 1:  # Copy to CPU temporary workspace
            # shape: (s, b * n_head, head_dim)
            k_buf, v_buf = dst.next_attention_compute_workspace()
            indices = (slice(0, self.task.prompt_len + i - 1),
                       slice(0, k_home.shape[1]))
            general_copy(k_buf, indices, k_home, indices)

            if self.policy.attn_sparsity >= 1.0:
                general_copy(v_buf, indices, v_home, indices)
                cache_read_buf.store(((k_buf, False), (v_buf, False)))
            else:
                #v_home : gpu/cpu    v_buf : cpu
                cache_read_buf.store(((k_buf, False), ((v_home, v_buf), False)))
        else:
            raise ValueError(f"Invalid path: {path}")
        
    def store_cache(self, cache_home: ValueHolder, cache_write_buf: ValueHolder, i):
        # shape: (s, b * n_head, head_dim)
        #这里的k_home, v_home 的长度应该为分配的最大长度，即是静态分配
        k_home, v_home = cache_home.val
        k_new, v_new = cache_write_buf.pop()

        if i == self.task.gen_len - 1:  # last token, no need to store cache
            return

        if i == 0:  # prefill
            indices = (slice(0, k_new.shape[0]),
                       slice(0, k_new.shape[1]))
        else:  # decoding
            #复制到对应索引位置
            pos = self.task.prompt_len + i
            indices = (slice(pos - k_new.shape[0], pos),
                       slice(0, k_new.shape[1]))

        general_copy(k_home, indices, k_new, None)
        general_copy(v_home, indices, v_new, None)

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype
    
    def forward(self, hidden: ValueHolder, cache_read_buf: ValueHolder, weight_read_buf: ValueHolder, attention_mask: ValueHolder,
                cache_write_buf: ValueHolder, i, k):  
        n_head = self.config.n_head
        num_key_value_heads = self.config.num_key_value_heads

        donate = [False] * 10
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((w_q, donate[2]), (w_k, donate[3]) ,
             (w_v, donate[4]), (w_out, donate[5]),
             (w_ln, donate[6]), (inv_freq, donate[7])) = weight_read_buf.pop()
        else:
            ((w_q, _), (w_k, _),
             (w_v, _), (w_out, _),
             (w_ln, _), (inv_freq, _)) = weight_read_buf.val

        if i == 0:  # prefill
            mask, donate[1] = attention_mask.val.smart_copy(self.compute)
            h, new_k_cache, new_v_cache = self.compute.mha(h, mask, w_q, 
                w_k, w_v, w_out, w_ln, inv_freq, n_head, num_key_value_heads, donate)
            cache_write_buf.store((new_k_cache, new_v_cache))
        else:  # decoding
            mask, donate[1] = attention_mask.val.smart_copy(self.attention_compute)
            (k_cache, donate[8]), (v_cache, donate[9]) = cache_read_buf.pop()
            h, new_k_cache, new_v_cache = self.compute.mha_gen(h, mask, w_q,
                 w_k, w_v, w_out, w_ln, inv_freq, n_head, num_key_value_heads,
                k_cache, v_cache, donate, self.policy.attn_sparsity)
            cache_write_buf.store((new_k_cache, new_v_cache))

        hidden.val = h

class MLP:
    def __init__(self, config: LlamaConfig, env: ExecutionEnv, policy: Policy, layer_id: int):
        self.config = config
        self.env = env
        self.layer_id = layer_id
        self.policy = policy
        self.compute = self.env.gpu
        self.weight_load_dst =  self.compute

        self.task = None

    def set_task(self, task):
        self.task = task

    def init_weight(self, weight_home, path):
        h, inter_h, dtype = (self.config.input_dim, self.config.ffn_embed_dim, self.config.dtype)
        path = os.path.join(path, '')
        ln_path = path + f"layers.{self.layer_id}.post_attention_layernorm.weight"
        path = path + f"layers.{self.layer_id}.mlp."
        weight_specs = [
            # gate_proj
            ((inter_h, h), dtype, path + "gate_proj.weight"),
            # up_proj
            ((inter_h, h), dtype, path + "up_proj.weight"),
            # down_proj
            ((h, inter_h), dtype, path + "down_proj.weight"),
            # w_ln
            ((h,), dtype, ln_path),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

    def load_weight(self, weight_home, weight_read_buf, k):
        gate, up, down, w_ln = weight_home.val
        if k == 0:
            dst1 = self.weight_load_dst
            dst2 = self.compute
            weight_read_buf.store((
                gate.smart_copy(dst1), up.smart_copy(dst1),
                down.smart_copy(dst1), w_ln.smart_copy(dst2)))

    def init_cache_one_gpu_batch(self, cache_home):
        pass  # do nothing

    def load_cache(self, cache_home, cache_read_buf, i):
        pass  # do nothing

    def store_cache(self, cache_home, cache_write_buf, i):
        pass  # do nothing

    def input_act_shape_and_dtype(self, batch_size, seq_len):
        return (batch_size, seq_len, self.config.input_dim), self.config.dtype

    def forward(self, hidden, cache_read_buf, weight_read_buf, attention_mask,
                cache_write_buf, i, k):
        donate = [False] * 5
        h, donate[0] = hidden.val, True

        if k == self.policy.num_gpu_batches - 1:
            # Clear the weight_read_buf if it is the last gpu batch
            ((gate, donate[1]), (up, donate[2]), 
             (down, donate[3]), (w_ln, donate[4])) = weight_read_buf.pop()
        else:
            ((gate, _), (up, _), (down, _), (w_ln, _)) = weight_read_buf.val

        h = self.compute.mlp(h, gate, up, down, w_ln, donate)
        hidden.val = h

class LlamaLM:
    def __init__(self,
                 config: Union[str, OptConfig],
                 env: ExecutionEnv,
                 path: str,
                 policy: Policy):
        if isinstance(config, str):
            config = get_opt_config(config)
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = policy.num_gpu_batches

        layers = []
        layers.append(InputEmbed(self.config, self.env, self.policy))
        for i in range(self.config.num_hidden_layers):
            layers.append(SelfAttention(self.config, self.env, self.policy, i))
            layers.append(MLP(self.config, self.env, self.policy, i))
        layers.append(OutputEmbed(self.config, self.env, self.policy))
        self.layers = layers
        self.num_layers = len(layers)

        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        else:
            raise NotImplementedError()

        # CUDA streams
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches

        # cache[j][k]
        self.cache_home = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_read_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        self.cache_write_buf = array_2d(num_layers, num_gpu_batches, ValueHolder)
        # weight[j]
        self.weight_read_buf = array_1d(num_layers, ValueHolder)
        # attention_mask[k]
        self.attention_mask = array_1d(num_gpu_batches, ValueHolder)

        self.task = None
        self.init_all_weights()

    def set_task(self, task):
        self.task = task
        for l in self.layers:
            l.set_task(task)

    def init_weight(self, j):
        expanded_path = os.path.abspath(os.path.expanduser(
            os.path.join(self.path, f"{self.config.name}-np")))
        check_path = os.path.join(expanded_path, "embed_tokens.weight")
        #print(expanded_path)
        #print(check_path)
        if not os.path.exists(check_path) :
            download_opt_weights(self.config.name, self.path)

        self.layers[j].init_weight(self.weight_home[j], expanded_path)

    def load_weight(self, i, j, k, overlap=True):
        # Handle corner cases
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load from weight_home to weight_read_buf
        if overlap:
            with torch.cuda.stream(self.load_weight_stream):
                self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)
        else:
            self.layers[j].load_weight(self.weight_home[j], self.weight_read_buf[j], k)

    def delete_weight(self, j, k):
        if k == 0:
            for x in self.weight_home[j].pop():
                if isinstance(x, ValueHolder):
                    for y in x.pop():
                        y.delete()
                else:
                    x.delete()

    def init_cache(self, j, k):
        self.layers[j].init_cache_one_gpu_batch(self.cache_home[j][k])

    def load_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        if i == 0:  # prefill, no cache
            return
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load from cache_home to cache_read_buf
        if overlap:
            with torch.cuda.stream(self.load_cache_stream):
                self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)
        else:
            self.layers[j].load_cache(self.cache_home[j][k], self.cache_read_buf[j][k], i)

    def store_cache(self, i, j, k, overlap=True):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return
        if i == self.task.gen_len - 1:  # last token, no need to store cache
            self.cache_write_buf[j][k].pop()
            return

        # Store cache_write_buf to cache_home
        # Delete cache_write_buf
        if overlap:
            with torch.cuda.stream(self.store_cache_stream):
                self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)
        else:
            self.layers[j].store_cache(self.cache_home[j][k], self.cache_write_buf[j][k], i)

    def delete_cache(self, j, k):
        v = self.cache_home[j][k].pop()
        if v:
            for x in v:
                x.delete()

    def load_hidden(self, i, j, k):
        # Handle corner cases
        if k == self.num_gpu_batches:
            k = 0
            j += 1
        if j == self.num_layers:
            j = 0
            i += 1
            if i == self.execute_gen_len:
                return

        # Load to hidden states buffers
        dst = self.layers[j].compute
        dst: TorchDevice
        if j == 0:
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            if i == 0:  # load from the input ids
                val = dst.allocate((gpu_batch_size, self.task.prompt_len), np.int32)
                val.load_from_np(self.output_ids[left:right, :self.task.prompt_len])
            else:  # load from the last generated token
                pos = self.task.prompt_len + i
                val = dst.allocate((gpu_batch_size, 1), np.int32)
                val.load_from_np(self.output_ids[left:right, pos-1:pos])
        else:  # load from the last layer
            val = self.hidden[i][j-1][k].pop().move(dst)
        self.hidden[i][j][k].store(val)

    def store_hidden(self, i, j, k):
        # Handle corner cases
        if k == -1:
            k = self.num_gpu_batches - 1
            j -= 1
        if j == -1:
            j = self.num_layers - 1
            i -= 1
            if i == -1:
                return

        # Store to hidden states buffers
        if j == self.num_layers - 1:  # store to output
            gpu_batch_size = self.policy.gpu_batch_size
            left, right = k * gpu_batch_size, (k + 1) * gpu_batch_size
            ids = self.hidden[i][j][k].pop().data.detach().cpu().numpy()
            pos = self.task.prompt_len + i
            if self.task.stop:
                stopped = self.stopped[left:right]
                self.output_ids[left:right, pos:pos+1] = np.where(
                    stopped, self.config.pad_token_id, ids)
                stopped[:] = np.logical_or(stopped, ids == self.task.stop)
            else:
                self.output_ids[left:right, pos:pos+1] = ids
        else:  # move to home
            x = self.hidden[i][j][k]
            if x.val:  # x may already be moved due to overlapping
                x.val = x.val.move(self.act_home)

    def compute_layer(self, i, j, k):
        # Update the hidden in place
        # Clear the weight_read_buf if it is the last gpu batch
        # Clear the cache_read_buf
        # Run layer computation
        self.layers[j].forward(self.hidden[i][j][k], self.cache_read_buf[j][k],
            self.weight_read_buf[j], self.attention_mask[k],
            self.cache_write_buf[j][k], i, k)

    def sync(self):
        self.env.disk.synchronize()
        torch.cuda.synchronize()

    def init_all_weights(self):
        self.weight_home = array_1d(self.num_layers, ValueHolder)
        for j in range(self.num_layers):
            self.init_weight(j)

    def delete_all_weights(self):
        for j in range(self.num_layers):
            self.delete_weight(j, 0)

    def update_attention_mask(self, i, k):
        if i > 0:
            mask = self.attention_mask[k]
            assert mask.val is not None
            mask.val = mask.val.device.extend_attention_mask(mask.val, [True])
            return

        gpu_batch_size = self.policy.gpu_batch_size
        left = k * gpu_batch_size
        right = left + gpu_batch_size
        input_ids = self.output_ids[left:right, :self.task.prompt_len]

        attention_compute = (self.env.cpu if self.policy.cpu_cache_compute
            else self.env.gpu)
        val = attention_compute.allocate(
            (self.policy.gpu_batch_size, self.task.prompt_len), bool)
        val.load_from_np((input_ids != self.config.pad_token_id))
        self.attention_mask[k].store(val)

    def generate(self,
                 inputs: Union[np.array, List[List[int]]],
                 max_new_tokens: int = 32,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 stop: Optional[int] = None,
                 debug_mode: Optional[str] = None,
                 cut_gen_len: Optional[int] = None,
                 verbose: int = 0):
        task = Task(
            inputs=inputs,
            prompt_len=len(inputs[0]),
            gen_len=max_new_tokens,
            cut_gen_len=cut_gen_len,
            do_sample=do_sample,
            temperature=temperature,
            stop=stop,
        )
        num_layers = self.num_layers
        num_gpu_batches = self.num_gpu_batches
        gpu_batch_size = self.policy.gpu_batch_size
        overlap = self.policy.overlap
        prompt_len, gen_len = task.prompt_len, task.gen_len
        self.execute_gen_len = task.cut_gen_len if task.cut_gen_len else task.gen_len

        # Output token ids
        self.output_ids = np.full((len(task.inputs), prompt_len + gen_len),
            self.config.pad_token_id, dtype=np.int32)
        self.stopped = np.zeros((len(task.inputs), 1), dtype=bool)
        self.output_ids[:, :prompt_len] = np.asarray(task.inputs)
        assert gpu_batch_size * num_gpu_batches == len(task.inputs)

        # Intermediate tensors
        # The following buffers store values used
        # for the i-th token, j-th layer, k-th gpu batch.
        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.cache_home[j][k].clear()
                self.cache_read_buf[j][k].clear()
                self.cache_write_buf[j][k].clear()
        for j in range(num_layers):
            self.weight_read_buf[j].clear()
        for k in range(num_gpu_batches):
            self.attention_mask[k].clear()
        self.hidden = array_3d(gen_len, num_layers, num_gpu_batches, ValueHolder)

        # Init cache
        self.set_task(task)
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.init_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.init_attention_compute_workspace(self.config, self.task, self.policy)
        self.generation_loop_normal()
        for j in range(num_layers):
            for k in range(num_gpu_batches):
                self.delete_cache(j, k)
        if self.policy.cpu_cache_compute:
            self.env.cpu.del_attention_compute_workspace()

        return self.output_ids

    def generation_loop_normal(self):
        for i in range(self.execute_gen_len):
            timers("generate").start()
            for k in range(self.num_gpu_batches):
                self.update_attention_mask(i, k)
            for j in range(self.num_layers):
                for k in range(self.num_gpu_batches):
                    self.load_weight(i, j, k, overlap=False)

                for k in range(self.num_gpu_batches):
                    self.load_cache(i, j, k, overlap=False)
                    self.load_hidden(i, j, k)
                    self.compute_layer(i, j, k)
                    self.store_hidden(i, j, k)
                    self.store_cache(i, j, k, overlap=False)
            timers("generate").stop()

    def __del__(self):
        self.delete_all_weights()

def get_filename(args):
    model_size = args.model.split('-')[-1]
    percent = ""
    for i in range(len(args.percent)):
        percent += str(args.percent[i]) + "-"
    filename = f"fo-{model_size}-gbs{args.gpu_batch_size}-" \
            f"ngbs{args.num_gpu_batches}-" \
            f"prompt{args.prompt_len}-" \
            f"gen{args.gen_len}-percent-{percent}"
    if args.cpu_cache_compute:
        filename += "cpu-cache"
    else:
        filename += "gpu-cache"
    if args.compress_weight:
        filename += "-compw"
    if args.compress_cache:
        filename += "-compc"
    return filename

def get_filename(args):
    model_size = args.model.split('-')[-1]
    percent = ""
    for i in range(len(args.percent)):
        percent += str(args.percent[i]) + "-"
    filename = f"fo-{model_size}-gbs{args.gpu_batch_size}-" \
               f"ngbs{args.num_gpu_batches}-" \
               f"prompt{args.prompt_len}-" \
               f"gen{args.gen_len}-percent-{percent}"
    if args.cpu_cache_compute:
        filename += "cpu-cache"
    else:
        filename += "gpu-cache"
    if args.compress_weight:
        filename += "-compw"
    if args.compress_cache:
        filename += "-compc"
    return filename
    
def get_test_inputs(prompt_len, num_prompts, tokenizer):
    prompts = ["Paris is the capital city of"]
    input_ids = tokenizer(prompts, padding="max_length",
                          max_length=prompt_len).input_ids
    return (input_ids[0],) * num_prompts


def run_flexgen(args):
    print(f"<run_flexgen>: args.model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.old_path, padding_side="left")
    tokenizer.pad_token = tokenizer.bos_token
    num_prompts = args.num_gpu_batches * args.gpu_batch_size
    prompt_len, gen_len, cut_gen_len = args.prompt_len, args.gen_len, args.cut_gen_len

    # Task and policy
    warmup_inputs = get_test_inputs(32, num_prompts, tokenizer)
    inputs = get_test_inputs(prompt_len, num_prompts, tokenizer)
    #print(warmup_inputs)

    gpu = TorchDevice("cuda:1")
    cpu = TorchDevice("cpu")
    env = ExecutionEnv(gpu=gpu, cpu=cpu)

    policy = Policy(args.gpu_batch_size, args.num_gpu_batches,
                    args.percent[0], args.percent[1],
                    args.percent[2], args.percent[3],
                    args.percent[4], args.percent[5],
                    args.overlap, args.sep_layer, args.pin_weight,
                    args.cpu_cache_compute, args.attn_sparsity,
                    args.compress_weight,
                    args.compress_cache)
    assert not (args.compress_cache and args.attn_sparsity < 1.0), "Not implemented"
    llama_config = LlamaConfig()
    cache_size = llama_config.cache_bytes(num_prompts, prompt_len + gen_len)
    hidden_size = llama_config.hidden_bytes(num_prompts, prompt_len + gen_len)
    print(f"model size: {llama_config.model_bytes()/GB:.3f} GB, "
          f"cache size: {cache_size/GB:.3f} GB, "
          f"hidden size (prefill): {hidden_size/GB:.3f} GB")

    print("init weight...")
    model = LlamaLM(llama_config, env, args.path, policy)

    try:
        print("warmup - generate")
        output_ids = model.generate(
            warmup_inputs, max_new_tokens=1, verbose=args.verbose)

        print("benchmark - generate")
        timers("generate").reset()
        output_ids = model.generate(
            inputs, max_new_tokens=args.gen_len,
            debug_mode=args.debug_mode, cut_gen_len=cut_gen_len, verbose=args.verbose)
        print(output_ids)
        costs = timers("generate").costs
    finally:
        pass

    # Log output
    prefill_latency = costs[0]
    prefill_throughput = num_prompts * prompt_len / prefill_latency
    if cut_gen_len:  # project latency of cut_gen_len to gen_len
        decode_latency = project_decode_latency(costs, prompt_len, gen_len)
    else:
        decode_latency = sum(costs[1:])
    decode_throughput = num_prompts * (gen_len - 1) / max(decode_latency, 1e-10)
    num_generated_tokens = num_prompts * gen_len
    total_latency = prefill_latency + decode_latency
    total_throughput = num_generated_tokens / total_latency
    _, gpu_peak_mem = gpu.mem_stats()
    _, cpu_peak_mem = cpu.mem_stats()

    if DUMMY_WEIGHT not in args.path:
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        show_str = "Outputs:\n" + 70 * '-' + "\n"
        for i in [0, len(outputs)-1]:
            show_str += f"{i}: {outputs[i]}\n"
            show_str += "-" * 70 + "\n"
        if args.verbose >= 2:
            print(show_str)

    gpu.print_stats()
    cpu.print_stats()
    projected = bool(args.debug_mode or cut_gen_len)

    if args.log_file == "auto":
        filename = get_filename(args) + ".log"
    else:
        filename = args.log_file

    log_str = write_benchmark_log(filename,
        llama_config.model_bytes(), cache_size, hidden_size,
        gpu_peak_mem, projected, prefill_latency, prefill_throughput,
        decode_latency, decode_throughput, total_latency, total_throughput)
    if args.verbose >= 1:
        print(log_str)

def add_parser_arguments(parser):
    parser.add_argument("--model", type=str, default="facebook/opt-6.7b",
        help="The model name.")
    parser.add_argument("--old-path", type=str, default="~/opt_weights",
        help="The path to the orginal model weights. ")
    parser.add_argument("--path", type=str, default="~/opt_weights",
        help="The path to the model weights. If there are no cached weights, "
             "FlexGen will automatically download them from HuggingFace.")
    parser.add_argument("--offload-dir", type=str, default="~/flexgen_offload_dir",
        help="The directory to offload tensors. ")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--gen-len", type=int, default=32)
    parser.add_argument("--cut-gen-len", type=int,
        help="Cut generation length for fast debugging.")
    parser.add_argument("--debug-mode", type=str,
        choices=["fewer_batch", "breakdown"])
    parser.add_argument("--gpu-batch-size", type=int, default=4)
    parser.add_argument("--num-gpu-batches", type=int, default=1)
    parser.add_argument("--percent", nargs="+", type=int,
        default=[100, 0, 100, 0, 100, 0],
        help="Six numbers. They are "
         "the percentage of weight on GPU, "
         "the percentage of weight on CPU, "
         "the percentage of attention cache on GPU, "
         "the percentage of attention cache on CPU, "
         "the percentage of activations on GPU, "
         "the percentage of activations on CPU")
    parser.add_argument("--sep-layer", type=str2bool, nargs='?',
        const=True, default=True)
    parser.add_argument("--pin-weight", type=str2bool, nargs="?",
        const=True, default=True)
    parser.add_argument("--cpu-cache-compute", action="store_true")
    parser.add_argument("--attn-sparsity", type=float, default=1.0)
    parser.add_argument("--compress-weight", action="store_true",
        help="Whether to compress weight.")
    parser.add_argument("--compress-cache", action="store_true",
        help="Whether to compress cache.")


    parser.add_argument("--log-file", type=str, default="auto")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)

    parser.add_argument("--overlap", type=str2bool, nargs='?',
        const=True, default=True)    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser)
    args = parser.parse_args()

    assert len(args.percent) == 6

    run_flexgen(args)
