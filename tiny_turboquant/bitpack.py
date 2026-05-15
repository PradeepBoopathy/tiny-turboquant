"""Bit-packing utilities for low-bit quantized index tensors.

The functions pack integer indices into real uint8 tensors. This is the
important difference between a real compressed cache and a cache that merely
*reports* a theoretical compressed size while storing int32 indices.

Supported bit widths: 1..8.
"""

from __future__ import annotations

from math import prod
from typing import Sequence, Tuple

import torch


Shape = Tuple[int, ...]


def _validate_bits(bits: int) -> None:
    if not 1 <= int(bits) <= 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")


def packed_num_bytes(n_values: int, bits: int) -> int:
    """Number of bytes needed to store ``n_values`` values at ``bits`` each."""
    _validate_bits(bits)
    return (int(n_values) * int(bits) + 7) // 8


def pack_indices(idx: torch.Tensor, bits: int) -> tuple[torch.Tensor, Shape]:
    """Pack an integer index tensor into a flat uint8 byte tensor.

    Args:
        idx: Integer tensor whose values must be in ``[0, 2**bits)``.
        bits: Number of bits per index, from 1 to 8.

    Returns:
        ``(packed, original_shape)`` where ``packed`` is a 1-D uint8 tensor.
    """
    _validate_bits(bits)
    if idx.numel() == 0:
        return torch.empty(0, device=idx.device, dtype=torch.uint8), tuple(idx.shape)

    max_val = (1 << bits) - 1
    if torch.any(idx < 0) or torch.any(idx > max_val):
        raise ValueError(f"indices must be in [0, {max_val}] for {bits}-bit packing")

    flat = idx.reshape(-1).to(torch.int64)
    n_values = flat.numel()
    n_bytes = packed_num_bytes(n_values, bits)

    # Use an integer accumulator wider than uint8 because scatter_add_ is used.
    out_i16 = torch.zeros(n_bytes, device=idx.device, dtype=torch.int16)
    positions = torch.arange(n_values, device=idx.device, dtype=torch.int64) * bits

    for j in range(bits):
        bit = (flat >> j) & 1
        bit_pos = positions + j
        byte_pos = bit_pos >> 3
        offset = bit_pos & 7
        payload = (bit << offset).to(torch.int16)
        out_i16.scatter_add_(0, byte_pos, payload)

    return out_i16.to(torch.uint8).contiguous(), tuple(idx.shape)


def unpack_indices(packed: torch.Tensor, bits: int, shape: Sequence[int]) -> torch.Tensor:
    """Unpack a uint8 byte tensor produced by :func:`pack_indices`."""
    _validate_bits(bits)
    shape = tuple(int(s) for s in shape)
    n_values = int(prod(shape)) if shape else 1
    if n_values == 0:
        return torch.empty(shape, device=packed.device, dtype=torch.uint8)

    expected_bytes = packed_num_bytes(n_values, bits)
    if packed.numel() < expected_bytes:
        raise ValueError(
            f"packed tensor too small: got {packed.numel()} bytes, expected {expected_bytes}"
        )

    byte_stream = packed.reshape(-1).to(torch.int64)
    values = torch.zeros(n_values, device=packed.device, dtype=torch.int64)
    positions = torch.arange(n_values, device=packed.device, dtype=torch.int64) * bits

    for j in range(bits):
        bit_pos = positions + j
        byte_pos = bit_pos >> 3
        offset = bit_pos & 7
        bit = (byte_stream.index_select(0, byte_pos) >> offset) & 1
        values |= bit << j

    return values.reshape(shape).to(torch.uint8)


def tensor_nbytes(x: torch.Tensor) -> int:
    return int(x.numel() * x.element_size())
