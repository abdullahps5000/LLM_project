"""
Retry logic with exponential backoff for network requests.
"""
from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable, TypeVar, Union

import httpx

T = TypeVar("T")


class RetryableError(Exception):
    """Base exception for retryable errors."""
    pass


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple[type[Exception], ...] = (
        httpx.ConnectError,
        httpx.TimeoutException,
        httpx.NetworkError,
        httpx.ReadError,
        ConnectionError,
        TimeoutError,
    ),
    retryable_status_codes: tuple[int, ...] = (500, 502, 503, 504),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        retryable_exceptions: Exceptions that trigger retry
        retryable_status_codes: HTTP status codes that trigger retry
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None
            delay = initial_delay
            
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    # Check if result is an HTTP response with retryable status
                    if isinstance(result, httpx.Response):
                        if result.status_code in retryable_status_codes:
                            if attempt < max_retries:
                                time.sleep(min(delay, max_delay))
                                delay *= exponential_base
                                continue
                            result.raise_for_status()
                    
                    return result
                    
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(min(delay, max_delay))
                        delay *= exponential_base
                    else:
                        raise
                except Exception as e:
                    # Non-retryable exception, raise immediately
                    raise
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError("Retry logic failed unexpectedly")
        
        return wrapper
    return decorator


def http_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
) -> Callable[[Callable[..., httpx.Response]], Callable[..., httpx.Response]]:
    """Specialized retry decorator for HTTP requests."""
    return retry_with_backoff(
        max_retries=max_retries,
        initial_delay=initial_delay,
        max_delay=max_delay,
        retryable_exceptions=(
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.ReadError,
            ConnectionError,
            TimeoutError,
        ),
        retryable_status_codes=(500, 502, 503, 504),
    )

