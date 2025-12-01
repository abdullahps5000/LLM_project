"""
Custom error classes with actionable error messages.
"""
from __future__ import annotations


class EBPError(Exception):
    """Base exception for EBP errors."""
    pass


class NetworkError(EBPError):
    """Network-related errors."""
    
    def __init__(self, message: str, url: str | None = None, suggestion: str | None = None):
        super().__init__(message)
        self.url = url
        self.suggestion = suggestion or self._default_suggestion()
    
    def _default_suggestion(self) -> str:
        if self.url:
            return (
                f"Check that the agent at {self.url} is running and reachable. "
                f"Try: curl {self.url}/v1/health"
            )
        return "Check network connectivity and agent status."


class MemoryError(EBPError):
    """Memory-related errors."""
    
    def __init__(
        self,
        message: str,
        required: int | None = None,
        available: int | None = None,
        suggestion: str | None = None,
    ):
        super().__init__(message)
        self.required = required
        self.available = available
        self.suggestion = suggestion or self._default_suggestion()
    
    def _default_suggestion(self) -> str:
        suggestions = []
        if self.required and self.available:
            from .common import human_bytes
            suggestions.append(
                f"Required: {human_bytes(self.required)}, Available: {human_bytes(self.available)}. "
            )
        suggestions.append(
            "Try: 1) Reduce --mem-fraction, 2) Use more devices, 3) Use a smaller model, "
            "4) Reduce context length"
        )
        return "".join(suggestions)


class PartitioningError(EBPError):
    """Partitioning-related errors."""
    
    def __init__(self, message: str, suggestion: str | None = None):
        super().__init__(message)
        self.suggestion = suggestion or (
            "Try: 1) Increase --mem-fraction, 2) Use more devices, "
            "3) Reduce --min-prefix, 4) Use a smaller model"
        )


class PackagingError(EBPError):
    """Packaging-related errors."""
    
    def __init__(self, message: str, suggestion: str | None = None):
        super().__init__(message)
        self.suggestion = suggestion or (
            "Try: 1) Reduce batch size, 2) Free up memory, "
            "3) Use --mem-fraction to reserve more memory, 4) Package on a machine with more RAM"
        )


class ModelError(EBPError):
    """Model-related errors."""
    
    def __init__(self, message: str, model_path: str | None = None, suggestion: str | None = None):
        super().__init__(message)
        self.model_path = model_path
        self.suggestion = suggestion or (
            "Ensure the model directory contains config.json and safetensors files. "
            "Check that the model is compatible with transformers library."
        )


class AgentError(EBPError):
    """Agent-related errors."""
    
    def __init__(self, message: str, agent_url: str | None = None, suggestion: str | None = None):
        super().__init__(message)
        self.agent_url = agent_url
        self.suggestion = suggestion or (
            "Check that the agent is running and accessible. "
            "Verify network connectivity and firewall settings."
        )


def format_error_with_suggestion(error: Exception) -> str:
    """Format an error with its suggestion if available."""
    if isinstance(error, EBPError) and hasattr(error, "suggestion"):
        return f"{str(error)}\n\nSuggestion: {error.suggestion}"
    return str(error)

