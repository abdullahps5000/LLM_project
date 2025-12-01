from __future__ import annotations

import argparse
import uuid

import uvicorn

from .agent_app import create_app, advertise
from .config import EBPConfig, get_config, set_config
from .logging_config import setup_logging


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--port", type=int, default=8008)
    p.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    p.add_argument("--log-level", type=str, default=None, help="Log level (DEBUG, INFO, WARNING, ERROR)")
    args = p.parse_args()
    
    # Load configuration
    config = EBPConfig.load(args.config)
    if args.log_level:
        config.logging.level = args.log_level
    set_config(config)
    
    # Setup logging
    setup_logging(
        level=config.logging.level,
        log_file=config.logging.log_file if config.logging.enable_file_logging else None,
        component=f"ebp.agent.{args.name}",
    )
    
    from .logging_config import get_logger
    logger = get_logger(f"ebp.agent.{args.name}")

    agent_id = uuid.uuid4().hex[:8]
    app = create_app(name=args.name, agent_id=agent_id, log_level=config.logging.level)
    advertise(name=args.name, port=args.port)

    logger.info(f"Agent {args.name} (id={agent_id}) starting on 0.0.0.0:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
