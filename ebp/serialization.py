"""
Binary tensor serialization for efficient network transfer.
"""
from __future__ import annotations

import struct
from typing import Tuple

import numpy as np
import torch

from .logging_config import get_logger

logger = get_logger("ebp.serialization")


def serialize_tensor(tensor: torch.Tensor) -> bytes:
    """
    Serialize a tensor to binary format.
    
    Format:
    - 4 bytes: dtype code (0=float32, 1=float16, 2=int64, 3=int32)
    - 4 bytes: num_dims
    - num_dims * 4 bytes: shape
    - tensor data (contiguous bytes)
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(tensor)}")
    
    # Ensure contiguous and on CPU
    tensor = tensor.contiguous().cpu()
    
    # Map dtype to code
    dtype_map = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 1,  # Treat as float16
        torch.int64: 2,
        torch.int32: 3,
    }
    
    dtype_code = dtype_map.get(tensor.dtype)
    if dtype_code is None:
        # Convert to float32 if unsupported
        logger.warning(f"Unsupported dtype {tensor.dtype}, converting to float32")
        tensor = tensor.float()
        dtype_code = 0
    
    # Serialize header
    num_dims = len(tensor.shape)
    header = struct.pack(">ii", dtype_code, num_dims)
    header += struct.pack(f">{num_dims}i", *tensor.shape)
    
    # Serialize data
    if tensor.dtype == torch.bfloat16:
        # Convert bfloat16 to bytes
        data = tensor.numpy().tobytes()
    else:
        data = tensor.numpy().tobytes()
    
    return header + data


def deserialize_tensor(data: bytes) -> torch.Tensor:
    """
    Deserialize a tensor from binary format.
    """
    if len(data) < 8:
        raise ValueError("Invalid tensor data: too short")
    
    # Read header
    dtype_code, num_dims = struct.unpack(">ii", data[:8])
    
    if num_dims < 0 or num_dims > 10:
        raise ValueError(f"Invalid num_dims: {num_dims}")
    
    # Read shape
    shape_bytes = data[8:8 + num_dims * 4]
    if len(shape_bytes) < num_dims * 4:
        raise ValueError("Invalid tensor data: shape incomplete")
    
    shape = struct.unpack(f">{num_dims}i", shape_bytes)
    data_start = 8 + num_dims * 4
    
    # Map code to dtype
    dtype_map = {
        0: torch.float32,
        1: torch.float16,
        2: torch.int64,
        3: torch.int32,
    }
    
    dtype = dtype_map.get(dtype_code, torch.float32)
    
    # Read tensor data
    tensor_data = data[data_start:]
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    
    # Calculate expected size
    dtype_size = {
        torch.float32: 4,
        torch.float16: 2,
        torch.int64: 8,
        torch.int32: 4,
    }.get(dtype, 4)
    
    expected_size = num_elements * dtype_size
    
    if len(tensor_data) < expected_size:
        raise ValueError(
            f"Invalid tensor data: expected {expected_size} bytes, got {len(tensor_data)}"
        )
    
    # Create numpy array and convert to tensor
    numpy_dtype = dtype_to_numpy(dtype)
    np_array = np.frombuffer(tensor_data[:expected_size], dtype=numpy_dtype)
    np_array = np_array.reshape(shape).copy()
    tensor = torch.from_numpy(np_array)
    
    return tensor


def dtype_to_numpy(dtype: torch.dtype):
    """Convert torch dtype to numpy dtype."""
    dtype_map = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.int64: np.int64,
        torch.int32: np.int32,
    }
    return dtype_map.get(dtype, np.float32)
