"""
Configuration management for EBP.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class NetworkConfig:
    """Network-related configuration."""
    timeout_s: float = 10.0
    stage_load_timeout_s: float = 900.0
    max_retries: int = 3
    retry_initial_delay: float = 1.0
    retry_max_delay: float = 60.0
    retry_exponential_base: float = 2.0


@dataclass
class MemoryConfig:
    """Memory-related configuration."""
    mem_fraction: float = 0.40  # Reduced from 0.45 for more safety margin
    batch_size_small: int = 1  # Reduced from 2 for maximum safety
    batch_size_normal: int = 2  # Reduced from 10 for maximum safety
    gc_interval: int = 1  # GC after every N batches
    kv_cache_reserve_fraction: float = 0.20  # Reserve 20% for KV cache
    min_free_memory_mb: int = 512  # Minimum free memory required (MB) before starting packaging
    memory_check_interval: int = 5  # Check memory every N batches


@dataclass
class ProfilingConfig:
    """Profiling-related configuration."""
    matmul_n: int = 1024
    matmul_iters: int = 5
    profile_timeout_s: float = 30.0


@dataclass
class PackagingConfig:
    """Packaging-related configuration."""
    dtype: str = "fp16"
    serve_port: int = 8090
    out_root: str = "./stages_out"


@dataclass
class LoggingConfig:
    """Logging-related configuration."""
    level: str = "INFO"
    log_file: Optional[str] = None
    enable_file_logging: bool = False


@dataclass
class EBPConfig:
    """Main EBP configuration."""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    packaging: PackagingConfig = field(default_factory=PackagingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EBPConfig":
        """Create config from dictionary."""
        config = cls()
        
        if "network" in d:
            config.network = NetworkConfig(**d["network"])
        if "memory" in d:
            config.memory = MemoryConfig(**d["memory"])
        if "profiling" in d:
            config.profiling = ProfilingConfig(**d["profiling"])
        if "packaging" in d:
            config.packaging = PackagingConfig(**d["packaging"])
        if "logging" in d:
            config.logging = LoggingConfig(**d["logging"])
        
        return config
    
    @classmethod
    def from_yaml(cls, path: str) -> "EBPConfig":
        """Load config from YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
        
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_env(cls) -> "EBPConfig":
        """Load config from environment variables."""
        config = cls()
        
        # Network
        if os.getenv("EBP_NETWORK_TIMEOUT"):
            config.network.timeout_s = float(os.getenv("EBP_NETWORK_TIMEOUT"))
        if os.getenv("EBP_MAX_RETRIES"):
            config.network.max_retries = int(os.getenv("EBP_MAX_RETRIES"))
        
        # Memory
        if os.getenv("EBP_MEM_FRACTION"):
            config.memory.mem_fraction = float(os.getenv("EBP_MEM_FRACTION"))
        if os.getenv("EBP_BATCH_SIZE"):
            config.memory.batch_size_normal = int(os.getenv("EBP_BATCH_SIZE"))
        
        # Logging
        if os.getenv("EBP_LOG_LEVEL"):
            config.logging.level = os.getenv("EBP_LOG_LEVEL")
        if os.getenv("EBP_LOG_FILE"):
            config.logging.log_file = os.getenv("EBP_LOG_FILE")
            config.logging.enable_file_logging = True
        
        return config
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "EBPConfig":
        """
        Load configuration from file, environment, or defaults.
        
        Priority: config file > environment > defaults
        """
        # Try config file first
        if config_path and Path(config_path).exists():
            try:
                return cls.from_yaml(config_path)
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
        
        # Try default config file
        default_config = Path.home() / ".ebp" / "config.yaml"
        if default_config.exists():
            try:
                return cls.from_yaml(str(default_config))
            except Exception:
                pass
        
        # Load from environment
        env_config = cls.from_env()
        
        # Merge with defaults (environment overrides)
        return env_config


# Global config instance
_config: Optional[EBPConfig] = None


def get_config() -> EBPConfig:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = EBPConfig.load()
    return _config


def set_config(config: EBPConfig) -> None:
    """Set global configuration instance."""
    global _config
    _config = config

