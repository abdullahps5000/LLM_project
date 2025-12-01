#!/usr/bin/env python3
"""
Run full inference with tokenization and generation.
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ebp.inference_engine import DistributedInferenceEngine
from ebp.logging_config import setup_logging, get_logger

logger = get_logger("run_inference")


def main():
    parser = argparse.ArgumentParser(description="Run distributed LLM inference")
    parser.add_argument("--plan", required=True, help="Path to plan.json")
    parser.add_argument("--prompt", required=True, help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=50, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding instead of sampling")
    parser.add_argument("--stream", action="store_true", help="Stream tokens as they're generated")
    parser.add_argument("--use-binary", action="store_true", default=True, help="Use binary protocol (faster)")
    parser.add_argument("--no-binary", action="store_true", help="Disable binary protocol (use JSON)")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    setup_logging(level=args.log_level, component="run_inference")
    
    logger.info("=" * 60)
    logger.info("Distributed LLM Inference")
    logger.info("=" * 60)
    
    # Initialize engine
    try:
        use_binary = not args.no_binary if hasattr(args, 'no_binary') else True
        engine = DistributedInferenceEngine(
            args.plan,
            use_binary=use_binary,
        )
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        sys.exit(1)
    
    # Generate
    try:
        generated_text = engine.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=not args.greedy,
            stream=args.stream,
        )
        
        if not args.stream:
            print("\n" + "=" * 60)
            print("Generated Text:")
            print("=" * 60)
            print(generated_text)
            print("=" * 60)
        else:
            print()  # Newline after streaming
        
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
