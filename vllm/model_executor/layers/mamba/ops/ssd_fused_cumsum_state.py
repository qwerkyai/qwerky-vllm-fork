# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Fused Cumsum + ChunkState kernel for Mamba2 SSD prefill.
#
# Replaces the two-kernel sequence:
#   dA_cumsum, dt = _chunk_cumsum_fwd(dt_raw, A, ...)
#   states = _chunk_state_fwd(B, x, dt, dA_cumsum, ...)
# with a single kernel that computes cumsum on-the-fly from raw dt,
# avoiding one kernel launch and the HBM read of dt_processed/dA_cumsum
# by the chunk_state kernel.
#
# dt_processed and dA_cumsum are still written to HBM (by the first tile
# of each chunk×head) because kernels 3 (state_passing) and 4+5 (fused_scan)
# need them.
#
# ruff: noqa: E501

import torch

from vllm.triton_utils import tl, triton

from .mamba_ssm import softplus


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=2,
        ),
    ],
    key=["hdim", "dstate", "chunk_size"],
)
@triton.jit
def _fused_cumsum_state_kernel(
    # Inputs
    x_ptr,                # (seqlen, nheads, hdim)
    b_ptr,                # (seqlen, ngroups, dstate)
    dt_raw_ptr,           # (seqlen, nheads) -- raw dt, NOT processed
    A_ptr,                # (nheads,)
    dt_bias_ptr,          # (nheads,) or None
    cu_chunk_seqlens_ptr, # (nchunks+1,)
    # Outputs
    states_ptr,           # (nchunks, nheads, hdim, dstate)
    dt_out_ptr,           # (nheads, nchunks, chunk_size) -- written by first tile only
    dA_cumsum_ptr,        # (nheads, nchunks, chunk_size) -- written by first tile only
    # Dimensions
    hdim: tl.constexpr,
    dstate: tl.constexpr,
    chunk_size: tl.constexpr,
    seqlen,
    nheads_ngroups_ratio: tl.constexpr,
    dt_min: tl.constexpr,
    dt_max: tl.constexpr,
    # Strides: x
    stride_x_seqlen: tl.int64,
    stride_x_head: tl.int64,
    stride_x_hdim: tl.constexpr,
    # Strides: b
    stride_b_seqlen: tl.int64,
    stride_b_head: tl.int64,
    stride_b_dstate: tl.constexpr,
    # Strides: dt_raw
    stride_dt_raw_seqlen: tl.int64,
    stride_dt_raw_head: tl.constexpr,
    # Strides: A
    stride_A_head: tl.constexpr,
    # Strides: dt_bias
    stride_dt_bias_head: tl.constexpr,
    # Strides: states
    stride_states_chunk: tl.int64,
    stride_states_head: tl.int64,
    stride_states_hdim: tl.int64,
    stride_states_dstate: tl.constexpr,
    # Strides: dt_out
    stride_dt_out_head: tl.int64,
    stride_dt_out_chunk: tl.int64,
    stride_dt_out_csize: tl.constexpr,
    # Strides: dA_cumsum
    stride_dA_cs_head: tl.int64,
    stride_dA_cs_chunk: tl.int64,
    stride_dA_cs_csize: tl.constexpr,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_c = tl.program_id(axis=1).to(tl.int64)   # chunk index
    pid_h = tl.program_id(axis=2)                  # head index
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n    # hdim tile
    pid_n = tl.program_id(axis=0) % num_pid_n     # dstate tile

    # Chunk bounds
    chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
    chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1)
    chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

    # Load A and dt_bias scalars for this head
    A_val = tl.load(A_ptr + pid_h * stride_A_head).to(tl.float32)
    dt_bias_val = tl.zeros((1,), dtype=tl.float32)
    if HAS_DT_BIAS:
        dt_bias_val = tl.load(dt_bias_ptr + pid_h * stride_dt_bias_head).to(tl.float32)

    # Raw dt pointer for this (chunk, head)
    dt_raw_base = dt_raw_ptr + chunk_seqlen_start * stride_dt_raw_seqlen + pid_h * stride_dt_raw_head

    # First pass: compute dA_cs_last (total cumulative decay)
    # Also optionally write dt_processed and dA_cumsum to HBM
    # (only first tile pid_m=0, pid_n=0 writes to avoid redundant writes)
    is_writer = (pid_m == 0) & (pid_n == 0)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dt_raw_ptrs_pass1 = dt_raw_base + offs_k * stride_dt_raw_seqlen
    running_sum = tl.zeros((1,), dtype=tl.float32)

    # Pointers for writing dt_out and dA_cumsum (only used by writer)
    dt_out_base = dt_out_ptr + pid_h * stride_dt_out_head + pid_c * stride_dt_out_chunk
    dA_cs_base = dA_cumsum_ptr + pid_h * stride_dA_cs_head + pid_c * stride_dA_cs_chunk
    dt_out_ptrs_w = dt_out_base + offs_k * stride_dt_out_csize
    dA_cs_ptrs_w = dA_cs_base + offs_k * stride_dA_cs_csize

    for k in range(0, chunk_size, BLOCK_SIZE_K):
        raw_dt = tl.load(
            dt_raw_ptrs_pass1,
            mask=offs_k < chunk_size_limit - k,
            other=0.0,
        ).to(tl.float32)

        # Apply softplus + bias + clamp
        if HAS_DT_BIAS:
            raw_dt += dt_bias_val
        if DT_SOFTPLUS:
            raw_dt = tl.where(raw_dt <= 20.0, softplus(raw_dt), raw_dt)
        dt_proc = tl.clamp(raw_dt, dt_min, dt_max)
        dt_proc = tl.where(offs_k < chunk_size_limit - k, dt_proc, 0.0)

        # Compute dA and local cumsum
        dA_block = dt_proc * A_val
        dA_cs_local = tl.cumsum(dA_block, axis=0)
        dA_cs_global = dA_cs_local + running_sum
        running_sum += tl.sum(dA_block, axis=0)

        # Writer tile stores dt_processed and dA_cumsum for kernels 3 and 4+5
        if is_writer:
            tl.store(
                dt_out_ptrs_w,
                dt_proc,
                mask=offs_k < chunk_size,
            )
            tl.store(
                dA_cs_ptrs_w,
                dA_cs_global,
                mask=offs_k < chunk_size,
            )
            dt_out_ptrs_w += BLOCK_SIZE_K * stride_dt_out_csize
            dA_cs_ptrs_w += BLOCK_SIZE_K * stride_dA_cs_csize

        dt_raw_ptrs_pass1 += BLOCK_SIZE_K * stride_dt_raw_seqlen

    dA_cs_last = running_sum

    # Main chunk_state computation
    # Same as original _chunk_state_fwd_kernel but computes dt/dA_cumsum
    # inline instead of reading from HBM.

    b_base = b_ptr + chunk_seqlen_start * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    x_base = x_ptr + chunk_seqlen_start * stride_x_seqlen + pid_h * stride_x_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    x_ptrs = x_base + (offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen)
    b_ptrs = b_base + (offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen)
    dt_raw_ptrs = dt_raw_base + offs_k * stride_dt_raw_seqlen

    running_sum2 = tl.zeros((1,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        # Inline cumsum: recompute dt_k and dA_cs_k from raw dt
        raw_dt = tl.load(
            dt_raw_ptrs,
            mask=offs_k < chunk_size_limit - k,
            other=0.0,
        ).to(tl.float32)

        if HAS_DT_BIAS:
            raw_dt += dt_bias_val
        if DT_SOFTPLUS:
            raw_dt = tl.where(raw_dt <= 20.0, softplus(raw_dt), raw_dt)
        dt_k = tl.clamp(raw_dt, dt_min, dt_max)
        dt_k = tl.where(offs_k < chunk_size_limit - k, dt_k, 0.0)

        dA_block = dt_k * A_val
        dA_cs_local = tl.cumsum(dA_block, axis=0)
        dA_cs_k = dA_cs_local + running_sum2
        running_sum2 += tl.sum(dA_block, axis=0)

        # Load x and B
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate),
            other=0.0,
        ).to(tl.float32)

        # Scale B by decay and dt (same as original)
        scale = tl.exp(dA_cs_last - dA_cs_k) * dt_k
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)

        # Advance pointers
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_raw_ptrs += BLOCK_SIZE_K * stride_dt_raw_seqlen

    # Store states
    states = acc.to(states_ptr.dtype.element_ty)
    states_out = states_ptr + pid_c * stride_states_chunk + pid_h * stride_states_head
    states_ptrs = states_out + (
        offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate
    )
    tl.store(states_ptrs, states, mask=(offs_m[:, None] < hdim) & (offs_n[None, :] < dstate))


def _fused_cumsum_chunk_state_fwd(
    dt_raw,
    A,
    B,
    x,
    chunk_size,
    cu_chunk_seqlens,
    dt_bias=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    states_in_fp32=True,
):
    """
    Fused Cumsum + ChunkState forward.

    Equivalent to:
        dA_cumsum, dt = _chunk_cumsum_fwd(dt_raw, A, chunk_size, cu_chunk_seqlens,
                                          dt_bias=dt_bias, dt_softplus=dt_softplus,
                                          dt_limit=dt_limit)
        states = _chunk_state_fwd(B, x, dt, dA_cumsum, cu_chunk_seqlens,
                                  states_in_fp32=states_in_fp32)

    Returns:
        dA_cumsum: (nheads, nchunks, chunk_size) float32
        dt_processed: (nheads, nchunks, chunk_size) float32
        states: (nchunks, nheads, hdim, dstate)
    """
    seqlen, nheads = dt_raw.shape
    _, nheads_x, hdim = x.shape
    _, ngroups, dstate = B.shape
    assert nheads == nheads_x
    assert nheads % ngroups == 0
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)

    nchunks = cu_chunk_seqlens.shape[0] - 1

    # Allocate outputs
    dt_out = torch.empty(
        nheads, nchunks, chunk_size, device=dt_raw.device, dtype=torch.float32
    )
    dA_cumsum = torch.empty(
        nheads, nchunks, chunk_size, device=dt_raw.device, dtype=torch.float32
    )
    states_dtype = torch.float32 if states_in_fp32 else B.dtype
    states = torch.empty(
        (nchunks, nheads, hdim, dstate), device=x.device, dtype=states_dtype
    )

    grid = lambda META: (
        triton.cdiv(hdim, META["BLOCK_SIZE_M"])
        * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),
        nchunks,
        nheads,
    )

    with torch.cuda.device(x.device.index):
        _fused_cumsum_state_kernel[grid](
            x_ptr=x,
            b_ptr=B,
            dt_raw_ptr=dt_raw,
            A_ptr=A,
            dt_bias_ptr=dt_bias,
            cu_chunk_seqlens_ptr=cu_chunk_seqlens,
            states_ptr=states,
            dt_out_ptr=dt_out,
            dA_cumsum_ptr=dA_cumsum,
            hdim=hdim,
            dstate=dstate,
            chunk_size=chunk_size,
            seqlen=seqlen,
            nheads_ngroups_ratio=nheads // ngroups,
            dt_min=dt_limit[0],
            dt_max=dt_limit[1],
            stride_x_seqlen=x.stride(0),
            stride_x_head=x.stride(1),
            stride_x_hdim=x.stride(2),
            stride_b_seqlen=B.stride(0),
            stride_b_head=B.stride(1),
            stride_b_dstate=B.stride(2),
            stride_dt_raw_seqlen=dt_raw.stride(0),
            stride_dt_raw_head=dt_raw.stride(1),
            stride_A_head=A.stride(0),
            stride_dt_bias_head=dt_bias.stride(0) if dt_bias is not None else 0,
            stride_states_chunk=states.stride(0),
            stride_states_head=states.stride(1),
            stride_states_hdim=states.stride(2),
            stride_states_dstate=states.stride(3),
            stride_dt_out_head=dt_out.stride(0),
            stride_dt_out_chunk=dt_out.stride(1),
            stride_dt_out_csize=dt_out.stride(2),
            stride_dA_cs_head=dA_cumsum.stride(0),
            stride_dA_cs_chunk=dA_cumsum.stride(1),
            stride_dA_cs_csize=dA_cumsum.stride(2),
            DT_SOFTPLUS=dt_softplus,
            HAS_DT_BIAS=dt_bias is not None,
        )

    return dA_cumsum, dt_out, states
