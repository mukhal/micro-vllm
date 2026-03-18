"""
Paged attention inference engine. vLLM-style batching with block-managed KV cache.
"""
import os
import glob
import json
from collections import deque
from copy import copy
from dataclasses import dataclass
from enum import Enum, auto
from itertools import count
from typing import List, Dict, Set, Optional

import numpy as np
import torch
import torch.nn.functional as F
import xxhash
from contextlib import nullcontext

from microvllm.model import GPT, GPTConfig
from microvllm.tokenizer import load_tokenizer


# -----------------------------------------------------------------------------
# Block manager: PagedAttention memory (from block_manager)
# -----------------------------------------------------------------------------

BLOCK_SIZE = 16


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 256
    top_k: Optional[int] = None
    ignore_eos: bool = False


class Sequence:
    block_size = BLOCK_SIZE
    counter = count()

    def __init__(self, token_ids: List[int], sampling_params: Optional[SamplingParams] = None):
        if sampling_params is None:
            sampling_params = SamplingParams()
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table: List[int] = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.top_k = sampling_params.top_k
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    def block(self, i: int) -> List[int]:
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size: (i + 1) * self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1


class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.token_ids: List[int] = []
        self.hash: int = -1

    def update(self, hash: int, token_ids: List[int]):
        self.hash = hash
        self.token_ids = token_ids
        self.ref_count += 1

    def reset(self):
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int = BLOCK_SIZE):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: Dict[int, int] = {}
        self.free_block_ids: deque = deque(range(num_blocks))
        self.used_block_ids: Set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: List[int], parent_hash: int = -1) -> int:
        h = xxhash.xxh64()
        if parent_hash != -1:
            h.update(parent_hash.to_bytes(8, 'big', signed=False))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> Block:
        if not self.free_block_ids:
            raise RuntimeError("Out of KV cache blocks")
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        block.reset()
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int):
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and block.hash in self.hash_to_block_id:
            if self.hash_to_block_id[block.hash] == block_id:
                del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.discard(block_id)
        self.free_block_ids.append(block_id)

    def allocate(self, seq: Sequence):
        h = -1
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id != -1 and self.blocks[block_id].token_ids == token_ids:
                if block_id in self.used_block_ids:
                    self.blocks[block_id].ref_count += 1
                else:
                    self.used_block_ids.add(block_id)
                    self.free_block_ids.remove(block_id)
                    self.blocks[block_id].ref_count = 1
            else:
                block = self._allocate_block()
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block.block_id
            seq.block_table.append(block.block_id)

    def deallocate(self, seq: Sequence):
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()


# -----------------------------------------------------------------------------
# Paged KV cache and sequence view
# -----------------------------------------------------------------------------


class PagedKVCache:
    def __init__(self, num_blocks, block_size, num_layers, num_heads, head_dim, device, dtype):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.k_cache = torch.zeros(
            num_blocks, num_layers, block_size, num_heads, head_dim,
            device=device, dtype=dtype
        )
        self.v_cache = torch.zeros(
            num_blocks, num_layers, block_size, num_heads, head_dim,
            device=device, dtype=dtype
        )

    def gather_kv_for_sequence(self, layer_idx, block_table, seq_len):
        num_full_blocks = seq_len // self.block_size
        remainder = seq_len % self.block_size
        k_parts = []
        v_parts = []
        for i in range(num_full_blocks):
            block_id = block_table[i]
            k_parts.append(self.k_cache[block_id, layer_idx])
            v_parts.append(self.v_cache[block_id, layer_idx])
        if remainder > 0:
            block_id = block_table[num_full_blocks]
            k_parts.append(self.k_cache[block_id, layer_idx, :remainder])
            v_parts.append(self.v_cache[block_id, layer_idx, :remainder])
        k = torch.cat(k_parts, dim=0).unsqueeze(0)
        v = torch.cat(v_parts, dim=0).unsqueeze(0)
        return k, v


class SequenceKVView:
    def __init__(self, paged_cache: PagedKVCache, block_table: List[int], seq_len: int, block_size: int):
        self.paged_cache = paged_cache
        self.block_table = block_table
        self.seq_len = seq_len
        self.cached_len = 0
        self.block_size = block_size
        self.n_layers = paged_cache.num_layers
        self.is_paged = True
        self.cache_seqlens = torch.tensor([0], dtype=torch.int32, device=paged_cache.device)

    def get_pos(self):
        return self.cached_len

    def get_layer_cache(self, layer_idx):
        if self.cached_len == 0:
            device = self.paged_cache.device
            dtype = self.paged_cache.dtype
            return (
                torch.empty(1, 0, self.paged_cache.num_heads, self.paged_cache.head_dim, device=device, dtype=dtype),
                torch.empty(1, 0, self.paged_cache.num_heads, self.paged_cache.head_dim, device=device, dtype=dtype),
            )
        return self.paged_cache.gather_kv_for_sequence(
            layer_idx, self.block_table, self.cached_len
        )

    def advance(self, num_tokens):
        self.cached_len += num_tokens
        self.cache_seqlens[0] = self.cached_len

    def write_kv(self, layer_idx, k, v):
        k = k.squeeze(0)
        v = v.squeeze(0)
        num_new_tokens = k.size(0)
        start_pos = self.cached_len
        for i in range(num_new_tokens):
            pos = start_pos + i
            block_idx = pos // self.block_size
            slot_idx = pos % self.block_size
            block_id = self.block_table[block_idx]
            self.paged_cache.k_cache[block_id, layer_idx, slot_idx] = k[i]
            self.paged_cache.v_cache[block_id, layer_idx, slot_idx] = v[i]


@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    assert temperature >= 0.0
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=rng)


# -----------------------------------------------------------------------------
# Scheduler
# -----------------------------------------------------------------------------


class Scheduler:
    def __init__(self, block_manager: BlockManager, max_batch_size: int = 32):
        self.block_manager = block_manager
        self.max_batch_size = max_batch_size
        self.waiting: List[Sequence] = []
        self.running: List[Sequence] = []

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple:
        if self.waiting:
            batch = []
            while self.waiting and len(batch) < self.max_batch_size:
                seq = self.waiting.pop(0)
                try:
                    self.block_manager.allocate(seq)
                    batch.append(seq)
                except RuntimeError:
                    # not enough blocks, put back in waiting queue!
                    self.waiting.insert(0, seq)
                    break
            if batch:
                return batch, True
        if self.running:
            return self.running, False
        return [], False

    def update(self, seqs: List[Sequence], new_tokens: List[int], stop_token_ids: Set[int]):
        for seq, token in zip(seqs, new_tokens):
            seq.append_token(token)
            if token in stop_token_ids:
                seq.status = SequenceStatus.FINISHED
            elif seq.num_completion_tokens >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED

    def finish_prefill(self, seqs: List[Sequence]):
        for seq in seqs:
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)

    def collect_finished(self) -> List[Sequence]:
        finished = [seq for seq in self.running if seq.is_finished]
        self.running = [seq for seq in self.running if not seq.is_finished]
        for seq in finished:
            self.block_manager.deallocate(seq)
        return finished

    def is_finished(self) -> bool:
        return len(self.waiting) == 0 and len(self.running) == 0


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


def _find_last_step(checkpoint_dir: str) -> int:
    pattern = os.path.join(checkpoint_dir, "model_*.pt")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in files))


def _patch_config(meta_data: dict) -> dict:
    kwargs = meta_data.get("model_config", {})
    if "window_pattern" not in kwargs:
        kwargs["window_pattern"] = "L"
    return kwargs


def _load_model(checkpoint_dir: str, step: Optional[int], device: torch.device, tokenizer_path: Optional[str]):
    if step is None:
        step = _find_last_step(checkpoint_dir)
    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    model_data = torch.load(model_path, map_location=device)
    if device.type in {"cpu", "mps"}:
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)
    config_kwargs = _patch_config(meta_data)
    config = GPTConfig(**config_kwargs)
    n_layer = config.n_layer
    if "resid_lambdas" not in model_data:
        model_data["resid_lambdas"] = torch.ones(n_layer)
    if "x0_lambdas" not in model_data:
        model_data["x0_lambdas"] = torch.zeros(n_layer)
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=False, assign=True)
    model.eval()
    tokenizer = load_tokenizer(tokenizer_path)
    assert tokenizer.get_vocab_size() == config.vocab_size
    return model, tokenizer


# -----------------------------------------------------------------------------
# LLM
# -----------------------------------------------------------------------------


class LLM:
    def __init__(
        self,
        model_path: str,
        step: Optional[int] = None,
        tokenizer_path: Optional[str] = None,
        device: str = "cuda",
        num_blocks: int = 2000,
        block_size: int = BLOCK_SIZE,
        max_batch_size: int = 32,
    ):
        self.device = torch.device(device)
        if tokenizer_path is None:
            tokenizer_path = os.path.join(os.path.dirname(model_path), "tokenizer")
            if not os.path.isdir(tokenizer_path):
                tokenizer_path = os.path.expanduser("~/.cache/nanochat/tokenizer")
        self.model, self.tokenizer = _load_model(
            model_path, step, self.device, tokenizer_path
        )
        m = self.model.config
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.paged_kv_cache = PagedKVCache(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=m.n_layer,
            num_heads=m.n_kv_head,
            head_dim=m.n_embd // m.n_head,
            device=self.device,
            dtype=dtype,
        )
        self.block_manager = BlockManager(num_blocks, block_size)
        self.scheduler = Scheduler(self.block_manager, max_batch_size)
        self.block_size = block_size
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        self.stop_token_ids: Set[int] = {assistant_end, bos}
        self.autocast_ctx = (
            torch.amp.autocast(device_type=device, dtype=dtype)
            if device == "cuda"
            else nullcontext()
        )

    def add_request(self, prompt, sampling_params: Optional[SamplingParams] = None):
        if sampling_params is None:
            sampling_params = SamplingParams()
        if isinstance(prompt, str):
            bos = self.tokenizer.get_bos_token_id()
            tokens = self.tokenizer.encode(prompt, prepend=bos)
        else:
            tokens = prompt
        seq = Sequence(tokens, sampling_params)
        self.scheduler.add(seq)
        return seq.seq_id

    @torch.inference_mode()
    def step(self) -> tuple:
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return [], 0
        with self.autocast_ctx:
            if is_prefill:
                self._run_prefill(seqs)
                self.scheduler.finish_prefill(seqs)
                num_tokens = sum(seq.num_prompt_tokens for seq in seqs)
            else:
                self._run_decode(seqs)
                num_tokens = -len(seqs)
        finished = self.scheduler.collect_finished()
        finished_outputs = [(seq.seq_id, seq.completion_token_ids) for seq in finished]
        return finished_outputs, num_tokens

    def _run_prefill(self, seqs: List[Sequence]):
        new_tokens = []
        for seq in seqs:
            kv_view = SequenceKVView(
                self.paged_kv_cache,
                seq.block_table,
                seq.num_tokens,
                self.block_size,
            )
            ids = torch.tensor([seq.token_ids], dtype=torch.long, device=self.device)
            logits = self.model.forward(ids, kv_cache=kv_view)
            logits = logits[:, -1, :]
            rng = torch.Generator(device=self.device)
            rng.manual_seed(42 + seq.seq_id)
            next_token = sample_next_token(logits, rng, seq.temperature, seq.top_k)
            new_tokens.append(next_token[0, 0].item())
            seq._kv_view = kv_view
        self.scheduler.update(seqs, new_tokens, self.stop_token_ids)

    def _run_decode(self, seqs: List[Sequence]):
        # Pre-allocate blocks needed for decode (since we'll write 1 more token)
        for seq in seqs:
            blocks_needed = (seq.num_tokens + 1 + self.block_manager.block_size - 1) // self.block_manager.block_size
            while len(seq.block_table) < blocks_needed:
                try:
                    new_block = self.block_manager._allocate_block()
                    seq.block_table.append(new_block.block_id)
                except RuntimeError:
                    seq.status = SequenceStatus.FINISHED
                    break
        
        new_tokens = []
        for seq in seqs:
            if seq.is_finished:
                continue
            kv_view = seq._kv_view
            last_token = seq.token_ids[-1]
            ids = torch.tensor([[last_token]], dtype=torch.long, device=self.device)
            logits = self.model.forward(ids, kv_cache=kv_view)[:, -1, :]
            rng = torch.Generator(device=self.device)
            rng.manual_seed(42 + seq.seq_id + seq.num_tokens)
            next_token = sample_next_token(logits, rng, seq.temperature, seq.top_k)
            new_tokens.append(next_token[0, 0].item())
        self.scheduler.update(seqs, new_tokens, self.stop_token_ids)

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
        use_tqdm: bool = True,
    ) -> List[dict]:
        if sampling_params is None:
            sampling_params = SamplingParams()
        if use_tqdm:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
            except ImportError:
                use_tqdm = False
        seq_ids = []
        for prompt in prompts:
            seq_id = self.add_request(prompt, sampling_params)
            seq_ids.append(seq_id)
        outputs = {}
        while not self.is_finished():
            finished_outputs, _ = self.step()
            for seq_id, token_ids in finished_outputs:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        if use_tqdm:
            pbar.close()
        results = []
        for seq_id in seq_ids:
            token_ids = outputs.get(seq_id, [])
            text = self.tokenizer.decode(token_ids)
            results.append({"text": text, "token_ids": token_ids})
        return results
