#!/usr/bin/env python3
"""
Unified script to run distributed LLM inference.
Handles coordinator setup if needed, then runs inference.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from ebp.inference_engine import DistributedInferenceEngine
from ebp.logging_config import setup_logging, get_logger

logger = get_logger("run")


def check_plan(plan_path: str) -> bool:
    """Check if plan.json exists and is valid."""
    if not os.path.exists(plan_path):
        return False
    
    try:
        with open(plan_path, "r") as f:
            plan = json.load(f)
        # Check required fields
        if "pipeline_order" not in plan or "layer_ranges" not in plan:
            return False
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run distributed LLM inference. Automatically runs coordinator if plan.json doesn't exist."
    )
    parser.add_argument("--plan", type=str, default="plan.json", help="Path to plan.json file")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=50, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding")
    parser.add_argument("--stream", action="store_true", help="Stream tokens as they're generated")
    parser.add_argument("--use-binary", action="store_true", default=True, help="Use binary protocol (faster)")
    parser.add_argument("--no-binary", action="store_true", help="Disable binary protocol (use JSON)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    
    # Coordinator options (used if plan doesn't exist)
    parser.add_argument("--model-path", type=str, help="Model path (required if plan doesn't exist)")
    parser.add_argument("--urls", type=str, help="Agent URLs (required if plan doesn't exist)")
    parser.add_argument("--pipeline-order", type=str, help="Pipeline order (required if plan doesn't exist)")
    parser.add_argument("--mem-fraction", type=float, default=0.40, help="Memory fraction per device")
    parser.add_argument("--ctx", type=int, default=512, help="Context length")
    parser.add_argument("--min-prefix", type=int, default=4, help="Minimum layers per device")
    parser.add_argument("--auto-coordinator", action="store_true", default=True, help="Auto-run coordinator if plan missing")
    
    args = parser.parse_args()
    
    setup_logging(level=args.log_level, component="run")
    
    # Check if plan exists
    plan_exists = check_plan(args.plan)
    
    if not plan_exists:
        logger.info("=" * 60)
        logger.info("Plan not found - running coordinator first")
        logger.info("=" * 60)
        
        if not args.auto_coordinator:
            logger.error("Plan not found and --auto-coordinator is disabled.")
            logger.error("Either:")
            logger.error("  1. Run coordinator manually: python -m ebp.coordinator_main ...")
            logger.error("  2. Use --auto-coordinator (default) and provide --model-path, --urls, --pipeline-order")
            sys.exit(1)
        
        # Validate coordinator arguments
        if not args.model_path:
            logger.error("--model-path is required when plan doesn't exist")
            sys.exit(1)
        if not args.urls:
            logger.error("--urls is required when plan doesn't exist")
            sys.exit(1)
        if not args.pipeline_order:
            logger.error("--pipeline-order is required when plan doesn't exist")
            sys.exit(1)
        
        # Run coordinator
        logger.info("Running coordinator to create plan...")
        import subprocess
        cmd = [
            sys.executable, "-m", "ebp.coordinator_main",
            "--model-path", args.model_path,
            "--urls", args.urls,
            "--pipeline-order", args.pipeline_order,
            "--mem-fraction", str(args.mem_fraction),
            "--ctx", str(args.ctx),
            "--min-prefix", str(args.min_prefix),
            "--package",
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        
        if result.returncode != 0:
            logger.error("Coordinator failed. Cannot proceed with inference.")
            sys.exit(1)
        
        # Verify plan was created
        if not check_plan(args.plan):
            logger.error(f"Coordinator completed but {args.plan} is invalid or missing")
            sys.exit(1)
        
        logger.info("✓ Plan created successfully")
        logger.info("")
    
    # Run inference
    logger.info("=" * 60)
    logger.info("Running Inference")
    logger.info("=" * 60)
    
    try:
        use_binary = not args.no_binary
        engine = DistributedInferenceEngine(args.plan, auto_load_stages=True, use_binary=use_binary)
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
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

