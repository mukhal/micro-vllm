# microvllm

This is minimal vLLM-style inference with **paged attention**: a small, readable engine for running LLM inference with block-managed KV cache. No training, no serving stack—just the core inference loop.

I liked [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) a lot, but found it to be a bit more complicated than I liked, especially for someone looking to deeply understand LLM inference and play with KV cache memory management. I built this around [Karpathy's nanochat](https://github.com/karpathy/nanochat), so the model logic is also as simple as it can be--also especially that NanoChat KV cache uses contiguous memory allocation. 

## Install

```bash
pip install torch tiktoken xxhash numpy
# optional: pip install tqdm datasets (for benchmarking)
```

## Quick Start

Simple test with a few prompts:

```bash
python run.py --model-dir models/nanochat-d34/ --tokenizer-dir models/tokenizer/
```

Benchmark on Alpaca dataset:

```bash
python test_alpaca.py --model-dir models/nanochat-d34/ \
    --num-prompts 512 \
    --max-tokens 64 \
    --tokenizer-dir models/tokenizer/
```

Example output (512 prompts, 64 tokens each, single A6000 RTX GPU, using [sdobson/nanochat](https://huggingface.co/sdobson/nanochat) model):
```
================================================================================
Benchmark Results
================================================================================
  Total prompts:        512
  Total output tokens:  31801
  Total time:           691.36s
  Throughput:           46.0 tokens/sec
  Latency per request:  1350.3ms
================================================================================
```

## Usage

Load a nanochat-format checkpoint (directory with `model_*.pt` and `meta_*.json`) and generate:

```python
from microvllm import LLM, SamplingParams

llm = LLM(model_path="./models/nanochat-d34")
outputs = llm.generate(
    ["Hello, how are you?", "What is 2+2?"],
    sampling_params=SamplingParams(temperature=0.7, max_tokens=100)
)
for output in outputs:
    print(output["text"])
```

**Tokenizer:** By default the tokenizer is loaded from `{model_path}/../tokenizer` or `~/.cache/nanochat/tokenizer`. Override with `tokenizer_path`:

```python
llm = LLM(model_path="./checkpoints/d34", tokenizer_path="/path/to/tokenizer")
```

**SamplingParams:** `temperature`, `max_tokens`, `top_k`, `ignore_eos`.

## Layout

```
microvllm/
├── __init__.py    # exports LLM, SamplingParams
├── model.py       # GPT (rotary, GQA, sliding window)
├── engine.py      # PagedKVCache, BlockManager, Scheduler, LLM, load_model
├── tokenizer.py   # load_tokenizer, Tokenizer (tiktoken from disk)
└── attention.py   # Flash Attention 3 / SDPA fallback
```

- **engine.py** — Paged attention: `PagedKVCache`, `SequenceKVView`, `BlockManager`, `Scheduler`, `LLM`, and inline `_load_model` / `_find_last_step`.
- **model.py** — Inference-only GPT (no optimizer, no training utilities).
- **tokenizer.py** — Loads a tiktoken encoding from a directory containing `tokenizer.pkl` (nanochat-style).

## Dependencies

- `torch`
- `tiktoken`
- `xxhash`
- `numpy`
- optional: `tqdm` for progress in `generate()`

## License

MIT
