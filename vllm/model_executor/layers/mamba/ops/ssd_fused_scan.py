# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Fused BMM + ChunkScan kernel for Mamba2 SSD prefill.
#
# Replaces the two-kernel sequence:
#   CB = _bmm_chunk_fwd(C, B, ...)          # writes (nchunks, ngroups, c, c) to HBM
#   _chunk_scan_fwd(CB, x, ...)             # reads CB from HBM
# with a single kernel that computes CB on-the-fly in registers/SRAM,
# eliminating the intermediate CB tensor from HBM entirely.
#
# Savings at 4K context, chunk_size=64, ngroups=24:
#   CB tensor = 64 * 24 * 64 * 64 * 4 bytes = ~25 MB  (write + read = ~50 MB)
#   Plus one fewer kernel launch.
#
# ruff: noqa: E501

from packaging import version

import torch
from vllm.triton_utils import tl, triton

TRITON_22 = version.parse(triton.__version__) >= version.parse("2.2.0")


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=2,
        ),
    ],
    key=["chunk_size", "hdim", "dstate"],
)
@triton.jit
def _fused_chunk_scan_fwd_kernel(
    # Inputs 
    x_ptr,            # (seqlen, nheads, hdim)
    B_ptr,            # (seqlen, ngroups, dstate)   -- replaces cb_ptr
    C_ptr,            # (seqlen, ngroups, dstate)
    z_ptr,            # (seqlen, nheads, hdim)  or None
    out_ptr,          # (seqlen, nheads, hdim)  preallocated, written in-place
    dt_ptr,           # (nheads, nchunks, chunk_size)
    dA_cumsum_ptr,    # (nheads, nchunks, chunk_size)
    seq_idx_ptr,      # (nchunks,)
    states_ptr,       # (nchunks, nheads, hdim, dstate)  from state_passing
    D_ptr,            # (nheads, hdim) or (nheads,) or None
    initstates_ptr,   # (batch, nheads, hdim, dstate) or None
    cu_chunk_seqlens_ptr,  # (nchunks+1,)
    # Dimensions 
    chunk_size: tl.constexpr,
    hdim: tl.constexpr,
    dstate: tl.constexpr,
    seqlen,
    nheads_ngroups_ratio: tl.constexpr,
    # Strides: x 
    stride_x_seqlen: tl.int64,
    stride_x_head: tl.int64,
    stride_x_hdim: tl.constexpr,
    # Strides: B 
    stride_B_seqlen: tl.int64,
    stride_B_head: tl.int64,
    stride_B_dstate: tl.constexpr,
    # Strides: C 
    stride_C_seqlen: tl.int64,
    stride_C_head: tl.int64,
    stride_C_dstate: tl.constexpr,
    # Strides: z 
    stride_z_seqlen: tl.int64,
    stride_z_head: tl.int64,
    stride_z_hdim: tl.constexpr,
    # Strides: out 
    stride_out_seqlen: tl.int64,
    stride_out_head: tl.int64,
    stride_out_hdim: tl.constexpr,
    # Strides: dt / dA_cumsum
    stride_dt_chunk: tl.int64,
    stride_dt_head: tl.int64,
    stride_dt_csize: tl.constexpr,
    stride_dA_cs_chunk: tl.int64,
    stride_dA_cs_head: tl.int64,
    stride_dA_cs_csize: tl.constexpr,
    # Strides: seq_idx 
    stride_seq_idx_chunk: tl.constexpr,
    # Strides: states
    stride_states_chunk: tl.int64,
    stride_states_head: tl.int64,
    stride_states_hdim: tl.int64,
    stride_states_dstate: tl.constexpr,
    # Strides: initstates
    stride_init_states_batch: tl.int64,
    stride_init_states_head: tl.int64,
    stride_init_states_hdim: tl.int64,
    stride_init_states_dstate: tl.constexpr,
    # Strides: D 
    stride_D_head: tl.constexpr,
    # Meta-parameters
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_INITSTATES: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,   # tiles over chunk_size (token rows)
    BLOCK_SIZE_N: tl.constexpr,   # tiles over hdim (output cols)
    BLOCK_SIZE_K: tl.constexpr,   # tiles over chunk_size (token cols in scan)
    BLOCK_SIZE_DSTATE: tl.constexpr,  # must cover dstate in one shot when <= 128
):
    # Program IDs
    pid_c = tl.program_id(axis=1).to(tl.int64)   # chunk index
    pid_h = tl.program_id(axis=2)                  # head index
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n    # row block (chunk token positions)
    pid_n = tl.program_id(axis=0) % num_pid_n     # col block (hdim)

    # Group index (B and C are shared across heads in a group) 
    pid_g = pid_h // nheads_ngroups_ratio

    # Chunk token range
    chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
    chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1)
    chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

    # Advance base pointers to this chunk / head 
    x_ptr   += chunk_seqlen_start * stride_x_seqlen   + pid_h * stride_x_head
    B_ptr   += chunk_seqlen_start * stride_B_seqlen   + pid_g * stride_B_head
    C_ptr   += chunk_seqlen_start * stride_C_seqlen   + pid_g * stride_C_head
    dt_ptr  += pid_c * stride_dt_chunk                + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk       + pid_h * stride_dA_cs_head

    # Row / col offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # output token rows
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # hdim cols
    offs_k = tl.arange(0, BLOCK_SIZE_K)                          # k (scan)
    offs_ds = tl.arange(0, BLOCK_SIZE_DSTATE)                    # dstate

    # seq_idx for this chunk and the previous chunk
    seq_idx_ptr += pid_c * stride_seq_idx_chunk
    seq_idx      = tl.load(seq_idx_ptr)
    seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_chunk, mask=pid_c >= 1, other=-1)

    # Select previous state source 
    if HAS_INITSTATES and (seq_idx != seq_idx_prev):
        # New sequence: use initial state for this sequence
        prev_states_ptr  = (initstates_ptr
                            + seq_idx * stride_init_states_batch
                            + pid_h   * stride_init_states_head)
        prev_states_hdim   = stride_init_states_hdim
        prev_states_dstate = stride_init_states_dstate
    else:
        # Continuing: use state from previous chunk
        prev_states_ptr  = (states_ptr
                            + (pid_c - 1) * stride_states_chunk
                            + pid_h       * stride_states_head)
        prev_states_hdim   = stride_states_hdim
        prev_states_dstate = stride_states_dstate

    # dA_cumsum for output rows
    dA_cs_m = tl.load(
        dA_cumsum_ptr + offs_m * stride_dA_cs_csize,
        mask=offs_m < chunk_size,
        other=0.0,
    ).to(tl.float32)
    scale_m = tl.exp(dA_cs_m)

    # Accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # State contribution: acc += C @ prev_states * scale_m
    # Load C for this row block: (BLOCK_SIZE_M, BLOCK_SIZE_DSTATE)
    C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_ds[None, :] * stride_C_dstate)
    if BLOCK_SIZE_DSTATE <= 128:
        # Fast path: load all of dstate in one shot (dstate=128 fits)
        C_block = tl.load(
            C_ptrs,
            mask=(offs_m[:, None] < chunk_size_limit) & (offs_ds[None, :] < dstate),
            other=0.0,
        )
        if not HAS_INITSTATES and (seq_idx != seq_idx_prev):
            prev_states = tl.zeros((BLOCK_SIZE_DSTATE, BLOCK_SIZE_N), dtype=C_ptr.dtype.element_ty)
        else:
            prev_states_ptrs = (prev_states_ptr
                                + offs_n[None, :] * prev_states_hdim
                                + offs_ds[:, None] * prev_states_dstate)
            prev_states = tl.load(
                prev_states_ptrs,
                mask=(offs_ds[:, None] < dstate) & (offs_n[None, :] < hdim),
                other=0.0,
            ).to(C_ptr.dtype.element_ty)
        acc = tl.dot(C_block, prev_states) * scale_m[:, None]
    else:
        # Slow path: loop over dstate in chunks of BLOCK_SIZE_K
        prev_states_ptrs = (prev_states_ptr
                            + offs_n[None, :] * prev_states_hdim
                            + offs_ds[:, None] * prev_states_dstate)
        for ds in range(0, dstate, BLOCK_SIZE_K):
            C_ds = tl.load(
                C_ptrs,
                mask=(offs_m[:, None] < chunk_size_limit) & (offs_ds[None, :] < dstate - ds),
                other=0.0,
            )
            if not HAS_INITSTATES and (seq_idx != seq_idx_prev):
                ps = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=C_ptr.dtype.element_ty)
            else:
                ps = tl.load(
                    prev_states_ptrs,
                    mask=(offs_ds[:, None] < dstate - ds) & (offs_n[None, :] < hdim),
                    other=0.0,
                ).to(C_ptr.dtype.element_ty)
            acc += tl.dot(C_ds, ps)
            C_ptrs += BLOCK_SIZE_K * stride_C_dstate
            prev_states_ptrs += BLOCK_SIZE_K * prev_states_dstate
        acc *= scale_m[:, None]

    # Fused BMM + scan inner loop
    # For each k-block (token columns within chunk):
    #   1. Load B[k, :dstate]   shape (BLOCK_SIZE_K, BLOCK_SIZE_DSTATE)
    #   2. cb = C_block @ B_k.T shape (BLOCK_SIZE_M, BLOCK_SIZE_K)   [inline BMM]
    #   3. Scale cb by exp-decay and dt
    #   4. Apply causal mask (lower-triangular within chunk)
    #   5. Load x[k, :hdim]     shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
    #   6. acc += cb @ x
    #
    # C_block must be reloaded if BLOCK_SIZE_DSTATE < dstate (handled below).
    # For dstate=128 with BLOCK_SIZE_DSTATE=128, C_block is already loaded above.

    K_MAX = min((pid_m + 1) * BLOCK_SIZE_M, chunk_size_limit)  # causal: only process k <= m

    B_ptrs  = B_ptr + (offs_k[:, None] * stride_B_seqlen + offs_ds[None, :] * stride_B_dstate)
    x_ptrs  = x_ptr + (offs_k[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize

    for k in range(0, K_MAX, BLOCK_SIZE_K):
        # Load dA_cumsum[k] and dt[k] 
        dA_cs_k = tl.load(
            dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0
        ).to(tl.float32)
        dt_k = tl.load(
            dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0
        ).to(tl.float32)

        # Inline BMM: cb = C_block @ B_k.T
        if BLOCK_SIZE_DSTATE <= 128:
            # C_block already loaded as (BLOCK_SIZE_M, BLOCK_SIZE_DSTATE)
            B_k = tl.load(
                B_ptrs,
                mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_ds[None, :] < dstate),
                other=0.0,
            ).to(C_ptr.dtype.element_ty)
            # cb shape: (BLOCK_SIZE_M, BLOCK_SIZE_K)
            cb = tl.dot(C_block, tl.trans(B_k))
        else:
            # Accumulate over dstate dimension for large dstate
            cb = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
            C_ptrs_inner = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_ds[None, :] * stride_C_dstate)
            B_ptrs_inner = B_ptr + ((k + offs_k)[:, None] * stride_B_seqlen + offs_ds[None, :] * stride_B_dstate)
            for ds in range(0, dstate, BLOCK_SIZE_K):
                C_ds = tl.load(
                    C_ptrs_inner,
                    mask=(offs_m[:, None] < chunk_size_limit) & (offs_ds[None, :] < dstate - ds),
                    other=0.0,
                ).to(tl.float32)
                B_ds = tl.load(
                    B_ptrs_inner,
                    mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_ds[None, :] < dstate - ds),
                    other=0.0,
                ).to(tl.float32)
                cb += tl.dot(C_ds, tl.trans(B_ds))
                C_ptrs_inner += BLOCK_SIZE_K * stride_C_dstate
                B_ptrs_inner += BLOCK_SIZE_K * stride_B_dstate
            cb = cb.to(C_ptr.dtype.element_ty)

        # Scale by decay and dt
        cb = cb.to(tl.float32) * tl.exp(dA_cs_m[:, None] - dA_cs_k[None, :]) * dt_k[None, :]

        # Causal mask: output token m can only attend to past tokens k <= m
        causal_mask = offs_m[:, None] >= (k + offs_k[None, :])
        cb = tl.where(causal_mask, cb, 0.0)

        cb = cb.to(x_ptr.dtype.element_ty)

        # Load x and accumulate
        x = tl.load(
            x_ptrs,
            mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < hdim),
            other=0.0,
        )
        acc += tl.dot(cb, x)

        # Advance pointers 
        B_ptrs         += BLOCK_SIZE_K * stride_B_seqlen
        x_ptrs         += BLOCK_SIZE_K * stride_x_seqlen
        dt_ptrs        += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    # D residual 
    if HAS_D:
        if D_HAS_HDIM:
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        x_residual = tl.load(
            x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim),
            mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim),
            other=0.0,
        ).to(tl.float32)
        acc += x_residual * D

    # z gate 
    if HAS_Z:
        z_ptr += chunk_seqlen_start * stride_z_seqlen + pid_h * stride_z_head
        z = tl.load(
            z_ptr + (offs_m[:, None] * stride_z_seqlen + offs_n[None, :] * stride_z_hdim),
            mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim),
            other=0.0,
        ).to(tl.float32)
        acc *= z * tl.sigmoid(z)

    # Write output 
    out_ptr += chunk_seqlen_start * stride_out_seqlen + pid_h * stride_out_head
    tl.store(
        out_ptr + (offs_m[:, None] * stride_out_seqlen + offs_n[None, :] * stride_out_hdim),
        acc,
        mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim),
    )


def _fused_chunk_scan_fwd(
    x,
    B,
    C,
    dt,
    dA_cumsum,
    states,
    cu_chunk_seqlens,
    out,
    seq_idx,
    D=None,
    z=None,
    initial_states=None,
):
    """
    Fused BMM + ChunkScan forward pass.

    Computes the same result as:
        CB = _bmm_chunk_fwd(C, B, chunk_size, cu_chunk_seqlens, output_dtype=float32)
        _chunk_scan_fwd(CB, x, dt, dA_cumsum, C, states, cu_chunk_seqlens, out, seq_idx, D, z, initial_states)

    but without materialising CB in HBM.

    Arguments:
        x:           (seqlen, nheads, hdim)
        B:           (seqlen, ngroups, dstate)
        C:           (seqlen, ngroups, dstate)
        dt:          (nheads, nchunks, chunk_size)
        dA_cumsum:   (nheads, nchunks, chunk_size)
        states:      (nchunks, nheads, hdim, dstate)  -- output of state_passing
        cu_chunk_seqlens: (nchunks+1,)
        out:         (seqlen, nheads, hdim)  preallocated, written in-place
        seq_idx:     (nchunks,)
        D:           (nheads, hdim) or (nheads,) or None
        z:           (seqlen, nheads, hdim) or None
        initial_states: (batch, nheads, hdim, dstate) or None
    """
    assert seq_idx is not None, "seq_idx required for varlen"

    seqlen, nheads, hdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (seqlen, ngroups, dstate)
    assert C.shape == B.shape
    assert dt.shape == (nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert states.shape == (nchunks, nheads, hdim, dstate)
    assert seq_idx.shape == (nchunks,)
    if D is not None:
        assert D.shape == (nheads, hdim) or D.shape == (nheads,)
    if z is not None:
        assert z.shape == x.shape

    grid = lambda META: (
        triton.cdiv(chunk_size, META["BLOCK_SIZE_M"]) * triton.cdiv(hdim, META["BLOCK_SIZE_N"]),
        nchunks,
        nheads,
    )

    z_strides = (z.stride(0), z.stride(1), z.stride(2)) if z is not None else (0, 0, 0)
    init_strides = (
        (initial_states.stride(0), initial_states.stride(1),
         initial_states.stride(2), initial_states.stride(3))
        if initial_states is not None else (0, 0, 0, 0)
    )

    with torch.cuda.device(x.device.index):
        _fused_chunk_scan_fwd_kernel[grid](
            x_ptr=x,
            B_ptr=B,
            C_ptr=C,
            z_ptr=z,
            out_ptr=out,
            dt_ptr=dt,
            dA_cumsum_ptr=dA_cumsum,
            seq_idx_ptr=seq_idx,
            states_ptr=states,
            D_ptr=D,
            initstates_ptr=initial_states,
            cu_chunk_seqlens_ptr=cu_chunk_seqlens,
            chunk_size=chunk_size,
            hdim=hdim,
            dstate=dstate,
            seqlen=seqlen,
            nheads_ngroups_ratio=nheads // ngroups,
            stride_x_seqlen=x.stride(0),
            stride_x_head=x.stride(1),
            stride_x_hdim=x.stride(2),
            stride_B_seqlen=B.stride(0),
            stride_B_head=B.stride(1),
            stride_B_dstate=B.stride(2),
            stride_C_seqlen=C.stride(0),
            stride_C_head=C.stride(1),
            stride_C_dstate=C.stride(2),
            stride_z_seqlen=z_strides[0],
            stride_z_head=z_strides[1],
            stride_z_hdim=z_strides[2],
            stride_out_seqlen=out.stride(0),
            stride_out_head=out.stride(1),
            stride_out_hdim=out.stride(2),
            stride_dt_chunk=dt.stride(1),
            stride_dt_head=dt.stride(0),
            stride_dt_csize=dt.stride(2),
            stride_dA_cs_chunk=dA_cumsum.stride(1),
            stride_dA_cs_head=dA_cumsum.stride(0),
            stride_dA_cs_csize=dA_cumsum.stride(2),
            stride_seq_idx_chunk=seq_idx.stride(0),
            stride_states_chunk=states.stride(0),
            stride_states_head=states.stride(1),
            stride_states_hdim=states.stride(2),
            stride_states_dstate=states.stride(3),
            stride_init_states_batch=init_strides[0],
            stride_init_states_head=init_strides[1],
            stride_init_states_hdim=init_strides[2],
            stride_init_states_dstate=init_strides[3],
            stride_D_head=D.stride(0) if D is not None else 0,
            HAS_D=D is not None,
            D_HAS_HDIM=D.dim() == 2 if D is not None else True,
            HAS_Z=z is not None,
            HAS_INITSTATES=initial_states is not None,
            IS_TRITON_22=TRITON_22,
            BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        )
