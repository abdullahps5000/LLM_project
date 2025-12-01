"""
Unit tests for retry logic.
"""
import time
from unittest.mock import Mock, patch

import httpx
import pytest

from ebp.retry import http_retry, retry_with_backoff


def test_retry_success():
    """Test that successful call doesn't retry."""
    call_count = 0
    
    @retry_with_backoff(max_retries=3)
    def success_func():
        nonlocal call_count
        call_count += 1
        return "success"
    
    result = success_func()
    assert result == "success"
    assert call_count == 1


def test_retry_on_exception():
    """Test retry on retryable exception."""
    call_count = 0
    
    @retry_with_backoff(max_retries=2, initial_delay=0.01)
    def failing_func():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Connection failed")
        return "success"
    
    result = failing_func()
    assert result == "success"
    assert call_count == 3


def test_retry_exhausted():
    """Test that retries are exhausted after max attempts."""
    call_count = 0
    
    @retry_with_backoff(max_retries=2, initial_delay=0.01)
    def always_fail():
        nonlocal call_count
        call_count += 1
        raise ConnectionError("Always fails")
    
    with pytest.raises(ConnectionError):
        always_fail()
    
    assert call_count == 3  # Initial + 2 retries


def test_non_retryable_exception():
    """Test that non-retryable exceptions don't retry."""
    call_count = 0
    
    @retry_with_backoff(max_retries=3, initial_delay=0.01)
    def raise_value_error():
        nonlocal call_count
        call_count += 1
        raise ValueError("Not retryable")
    
    with pytest.raises(ValueError):
        raise_value_error()
    
    assert call_count == 1  # No retries

