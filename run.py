"""
Simple test script for microvllm.
"""

import argparse

from microvllm import LLM, SamplingParams


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=str, required=True, help="Checkpoint dir containing model_*.pt and meta_*.json")
    p.add_argument("--step", type=int, default=None, help="Checkpoint step (default: latest)")
    p.add_argument("--device", type=str, default="cuda", help="cuda|cpu|mps")
    p.add_argument("--tokenizer-dir", type=str, default=None, help="Dir containing tokenizer.pkl")
    p.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--num-blocks", type=int, default=2000, help="Number of KV cache blocks")
    p.add_argument("--max-batch-size", type=int, default=32, help="Max batch size for inference")
    args = p.parse_args()

    top_k = None if args.top_k <= 0 else args.top_k

    # Initialize LLM
    print(f"Initializing LLM from {args.model_dir}...")
    llm = LLM(
        model_path=args.model_dir,
        step=args.step,
        tokenizer_path=args.tokenizer_dir,
        device=args.device,
        num_blocks=args.num_blocks,
        max_batch_size=args.max_batch_size,
    )
    print(f"Model loaded on {args.device}")

    # Simple test prompts
    prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about the ocean.",
    ]

    # Generate
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_k=top_k,
    )

    print(f"\nGenerating responses for {len(prompts)} prompts...")
    print(f"  temperature={args.temperature}, max_tokens={args.max_tokens}, top_k={top_k}")
    print("=" * 80)

    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

    # Display results
    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt}")
        print(f"Output: {output['text']}")
        print(f"Tokens: {len(output['token_ids'])}")
        print("-" * 80)


if __name__ == "__main__":
    main()
