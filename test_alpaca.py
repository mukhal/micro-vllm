"""
Benchmark microvllm throughput on Alpaca dataset.
"""

import argparse
import time

import torch

from microvllm.engine import LLM, SamplingParams


def load_alpaca_instructions(n=512):
    """Load first n instructions from Alpaca dataset."""
    try:
        from datasets import load_dataset
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        prompts = []
        for i, example in enumerate(dataset):
            if i >= n:
                break
            instruction = example["instruction"]
            inp = example.get("input", "")
            if inp:
                prompt = f"{instruction}\n\n{inp}"
            else:
                prompt = instruction
            prompts.append(prompt)
        return prompts
    except Exception as e:
        print(f"Failed to load Alpaca dataset: {e}")
        print("Please install: pip install datasets")
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=str, required=True, help="Checkpoint dir containing model_*.pt and meta_*.json")
    p.add_argument("--step", type=int, default=None, help="Checkpoint step (default: latest)")
    p.add_argument("--device", type=str, default="cuda", help="cuda|cpu|mps")
    p.add_argument("--tokenizer-dir", type=str, default=None, help="Dir containing tokenizer.pkl")
    p.add_argument("--max-tokens", type=int, default=64, help="Max tokens to generate per request")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--num-prompts", type=int, default=512, help="Number of Alpaca prompts to use")
    p.add_argument("--num-blocks", type=int, default=4096, help="Number of KV cache blocks")
    p.add_argument("--max-batch-size", type=int, default=32, help="Max batch size for inference")
    args = p.parse_args()

    device = torch.device(args.device)
    top_k = None if args.top_k <= 0 else args.top_k

    print("=" * 80)
    print("microvllm Throughput Benchmark on Alpaca Dataset")
    print("=" * 80)

    # Load Alpaca prompts
    print(f"\nLoading {args.num_prompts} prompts from Alpaca dataset...")
    prompts = load_alpaca_instructions(args.num_prompts)
    print(f"Loaded {len(prompts)} prompts")
    
    # Show sample prompts
    print("\nSample prompts:")
    for i, p in enumerate(prompts[:3]):
        p_short = p[:100] + "..." if len(p) > 100 else p
        print(f"  [{i}] {p_short!r}")

    # Initialize LLM
    print(f"\nInitializing LLM from {args.model_dir} (step={args.step}) on {device}...")
    llm = LLM(
        model_path=args.model_dir,
        step=args.step,
        tokenizer_path=args.tokenizer_dir,
        device=str(device),
        num_blocks=args.num_blocks,
        max_batch_size=args.max_batch_size,
    )
    print(f"  num_blocks={args.num_blocks}, max_batch_size={args.max_batch_size}")

    # Run benchmark
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_k=top_k,
    )

    print("\n" + "=" * 80)
    print(f"Running benchmark: {len(prompts)} prompts, max_tokens={args.max_tokens}")
    print("=" * 80)

    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
    torch.cuda.synchronize() if device.type == "cuda" else None
    total_time = time.time() - t0

    # Compute metrics
    total_output_tokens = sum(len(o["token_ids"]) for o in outputs)
    throughput = total_output_tokens / total_time
    latency_per_request = total_time / len(prompts)

    print("\n" + "=" * 80)
    print("Benchmark Results")
    print("=" * 80)
    print(f"  Total prompts:        {len(prompts)}")
    print(f"  Total output tokens:  {total_output_tokens}")
    print(f"  Total time:           {total_time:.2f}s")
    print(f"  Throughput:           {throughput:.1f} tokens/sec")
    print(f"  Latency per request:  {latency_per_request*1000:.1f}ms")
    print("=" * 80)

    # Show sample outputs
    print("\nSample outputs:")
    for i, (prompt, output) in enumerate(zip(prompts[:3], outputs[:3])):
        prompt_short = prompt[:60] + "..." if len(prompt) > 60 else prompt
        text = output["text"]
        text_short = text[:120] + "..." if len(text) > 120 else text
        print(f"\n  [{i}] Prompt: {prompt_short!r}")
        print(f"      Output: {text_short!r}")
        print(f"      Tokens: {len(output['token_ids'])}")

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
