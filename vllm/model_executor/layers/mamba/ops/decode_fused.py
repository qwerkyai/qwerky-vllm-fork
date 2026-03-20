# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Fused Mamba2 decode kernel: causal_conv1d_update + selective_state_update in one launch.
#
# Restrictions (initial version):
#   - seqlen=1 per request (standard decode, no spec-decode)
#   - TIE_HDIM=True  (A, dt, dt_bias, D are scalar per head — always true for Mamba2)
#   - KERNEL_WIDTH=4 (d_conv=4, the Mamba2 default)
#   - No APC / prefix-caching (is_mamba_cache_all=False)

import torch
from packaging import version

from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

TRITON3 = HAS_TRITON and (version.parse(triton.__version__) >= version.parse("3.0.0"))

if TRITON3:

    @triton.jit
    def _softplus(dt):
        dt = tl.where(dt <= 20.0, tl.math.log(tl.math.exp(dt) + 1), dt)
        return dt

else:

    @triton.jit
    def _softplus(dt):
        dt = tl.where(dt <= 20.0, tl.math.log1p(tl.exp(dt)), dt)
        return dt


@triton.jit
def _mamba2_fused_decode_kernel(
    # Inputs
    x_bc_ptr,                   # (batch, conv_dim)  stride (conv_dim, 1)
    conv_state_ptr,             # (num_slots, conv_dim, state_len)  strides below
    w_ptr,                      # (conv_dim, kernel_width)
    bias_ptr,                   # (conv_dim,) or None
    conv_state_indices_ptr,     # (batch,) int32  – slot index per request
    # SSM weights (scalar per head: TIE_HDIM=True)
    ssm_state_ptr,              # (num_slots, nheads, head_dim, dstate)
    dt_ptr,                     # (batch, nheads)  unexpanded
    dt_bias_ptr,                # (nheads,)
    A_ptr,                      # (nheads,)
    D_ptr,                      # (nheads,)
    ssm_in_indices_ptr,         # (batch,) int32
    ssm_out_indices_ptr,        # (batch,) int32
    pad_slot_id,
    # Output
    out_ptr,                    # (batch, nheads, head_dim)
    # Scalar dims
    batch,
    # Constexpr dims
    nheads:     tl.constexpr,
    head_dim:   tl.constexpr,
    dstate:     tl.constexpr,
    n_groups:   tl.constexpr,
    d_inner:    tl.constexpr,   # = nheads * head_dim
    # state_len = KERNEL_WIDTH - 1
    KERNEL_WIDTH: tl.constexpr,  # must be 4 (d_conv=4)
    # Strides
    # x_bc: (batch, conv_dim), conv_dim is contiguous
    stride_xbc_batch,           # = conv_dim
    # conv_state: (slots, conv_dim, state_len)
    #   kv_cache[0] is (slots, state_len, conv_dim); after .transpose(-1,-2):
    #   strides become (state_len*conv_dim, 1, conv_dim)
    stride_cs_slot,             # = state_len * conv_dim
    stride_cs_dim,              # = 1  (conv_dim is contiguous)
    stride_cs_tok,              # = conv_dim
    # weights: (conv_dim, kernel_width), kernel_width contiguous
    stride_w_dim,               # = kernel_width
    stride_w_width,             # = 1
    # ssm_state: (slots, nheads, head_dim, dstate)
    stride_ss_slot,
    stride_ss_head,
    stride_ss_dim,
    stride_ss_dstate,
    # dt: (batch, nheads)
    stride_dt_batch,
    stride_dt_head,
    # out: (batch, nheads, head_dim), head_dim contiguous
    stride_out_batch,
    stride_out_head,
    stride_out_dim,             # = 1
    # Meta
    HAS_BIAS:   tl.constexpr,
    BLOCK_HD:   tl.constexpr,  # tile covering head_dim  (= next_pow2(head_dim))
    BLOCK_DS:   tl.constexpr,  # tile covering dstate    (= next_pow2(dstate))
):
    """
    One thread block per (batch, head).  Computes:
      1. conv1d update for x   (BLOCK_HD features, index range [h*head_dim, (h+1)*head_dim))
      2. conv1d update for B   (BLOCK_DS features, index range [d_inner + g*dstate, ...))
      3. conv1d update for C   (BLOCK_DS features, index range [d_inner + n_groups*dstate + g*dstate, ...))
      4. SSM step (TIE_HDIM): state = state*dA + B*dt*x;  y = state@C + x*D
    B and C are computed redundantly by every head in a group (safe: same values, idempotent writes).
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_b >= batch:
        return

    # Slot indices 
    conv_slot = tl.load(conv_state_indices_ptr + pid_b).to(tl.int64)
    if conv_slot == pad_slot_id:
        return

    ssm_slot_in  = tl.load(ssm_in_indices_ptr  + pid_b).to(tl.int64)
    ssm_slot_out = tl.load(ssm_out_indices_ptr  + pid_b).to(tl.int64)

    # Group for this head (for B/C indexing)
    nheads_per_group: tl.constexpr = nheads // n_groups
    g = pid_h // nheads_per_group

    # Helper: conv1d update for a block of features 
    # Returns acc (float32 vector), updates conv_state in HBM.
    # Inlined twice below (for x) and twice more (for B, C).

    # Conv1d for x  (head_dim features)
    hd_offs = tl.arange(0, BLOCK_HD)           # [BLOCK_HD]
    x_feat  = pid_h * head_dim + hd_offs       # feature indices in conv_dim
    mask_hd = hd_offs < head_dim

    # Base pointer into conv_state for these features at this slot
    cs_x = conv_state_ptr + conv_slot * stride_cs_slot + x_feat * stride_cs_dim

    # Load state_len=3 previous tokens (KERNEL_WIDTH=4)
    s0_x = tl.load(cs_x + 0 * stride_cs_tok, mask=mask_hd, other=0.0).to(tl.float32)
    s1_x = tl.load(cs_x + 1 * stride_cs_tok, mask=mask_hd, other=0.0).to(tl.float32)
    s2_x = tl.load(cs_x + 2 * stride_cs_tok, mask=mask_hd, other=0.0).to(tl.float32)

    # Load new input token
    new_x = tl.load(
        x_bc_ptr + pid_b * stride_xbc_batch + x_feat,
        mask=mask_hd, other=0.0,
    ).to(tl.float32)

    # Write shifted conv state: [s1, s2, new_x]
    tl.store(cs_x + 0 * stride_cs_tok, s1_x.to(conv_state_ptr.dtype.element_ty), mask=mask_hd)
    tl.store(cs_x + 1 * stride_cs_tok, s2_x.to(conv_state_ptr.dtype.element_ty), mask=mask_hd)
    tl.store(cs_x + 2 * stride_cs_tok, new_x.to(conv_state_ptr.dtype.element_ty), mask=mask_hd)

    # Dot product with conv weights
    w_x   = w_ptr + x_feat * stride_w_dim
    acc_x = (s0_x  * tl.load(w_x + 0 * stride_w_width, mask=mask_hd, other=0.0)
           + s1_x  * tl.load(w_x + 1 * stride_w_width, mask=mask_hd, other=0.0)
           + s2_x  * tl.load(w_x + 2 * stride_w_width, mask=mask_hd, other=0.0)
           + new_x * tl.load(w_x + 3 * stride_w_width, mask=mask_hd, other=0.0))
    if HAS_BIAS:
        acc_x = acc_x + tl.load(bias_ptr + x_feat, mask=mask_hd, other=0.0)

    # SiLU
    x_out = acc_x * tl.sigmoid(acc_x)  # [BLOCK_HD]

    # Conv1d for B  (dstate features for group g)
    ds_offs = tl.arange(0, BLOCK_DS)                         # [BLOCK_DS]
    b_feat  = d_inner + g * dstate + ds_offs
    mask_ds = ds_offs < dstate

    cs_b  = conv_state_ptr + conv_slot * stride_cs_slot + b_feat * stride_cs_dim

    s0_b  = tl.load(cs_b + 0 * stride_cs_tok, mask=mask_ds, other=0.0).to(tl.float32)
    s1_b  = tl.load(cs_b + 1 * stride_cs_tok, mask=mask_ds, other=0.0).to(tl.float32)
    s2_b  = tl.load(cs_b + 2 * stride_cs_tok, mask=mask_ds, other=0.0).to(tl.float32)
    new_b = tl.load(
        x_bc_ptr + pid_b * stride_xbc_batch + b_feat,
        mask=mask_ds, other=0.0,
    ).to(tl.float32)

    # All heads in a group write the same values → idempotent, safe
    tl.store(cs_b + 0 * stride_cs_tok, s1_b.to(conv_state_ptr.dtype.element_ty), mask=mask_ds)
    tl.store(cs_b + 1 * stride_cs_tok, s2_b.to(conv_state_ptr.dtype.element_ty), mask=mask_ds)
    tl.store(cs_b + 2 * stride_cs_tok, new_b.to(conv_state_ptr.dtype.element_ty), mask=mask_ds)

    w_b   = w_ptr + b_feat * stride_w_dim
    acc_b = (s0_b  * tl.load(w_b + 0 * stride_w_width, mask=mask_ds, other=0.0)
           + s1_b  * tl.load(w_b + 1 * stride_w_width, mask=mask_ds, other=0.0)
           + s2_b  * tl.load(w_b + 2 * stride_w_width, mask=mask_ds, other=0.0)
           + new_b * tl.load(w_b + 3 * stride_w_width, mask=mask_ds, other=0.0))
    if HAS_BIAS:
        acc_b = acc_b + tl.load(bias_ptr + b_feat, mask=mask_ds, other=0.0)

    b_out = acc_b * tl.sigmoid(acc_b)  # [BLOCK_DS]  SiLU

    # Conv1d for C  (dstate features for group g)
    c_feat = d_inner + n_groups * dstate + g * dstate + ds_offs

    cs_c  = conv_state_ptr + conv_slot * stride_cs_slot + c_feat * stride_cs_dim

    s0_c  = tl.load(cs_c + 0 * stride_cs_tok, mask=mask_ds, other=0.0).to(tl.float32)
    s1_c  = tl.load(cs_c + 1 * stride_cs_tok, mask=mask_ds, other=0.0).to(tl.float32)
    s2_c  = tl.load(cs_c + 2 * stride_cs_tok, mask=mask_ds, other=0.0).to(tl.float32)
    new_c = tl.load(
        x_bc_ptr + pid_b * stride_xbc_batch + c_feat,
        mask=mask_ds, other=0.0,
    ).to(tl.float32)

    tl.store(cs_c + 0 * stride_cs_tok, s1_c.to(conv_state_ptr.dtype.element_ty), mask=mask_ds)
    tl.store(cs_c + 1 * stride_cs_tok, s2_c.to(conv_state_ptr.dtype.element_ty), mask=mask_ds)
    tl.store(cs_c + 2 * stride_cs_tok, new_c.to(conv_state_ptr.dtype.element_ty), mask=mask_ds)

    w_c   = w_ptr + c_feat * stride_w_dim
    acc_c = (s0_c  * tl.load(w_c + 0 * stride_w_width, mask=mask_ds, other=0.0)
           + s1_c  * tl.load(w_c + 1 * stride_w_width, mask=mask_ds, other=0.0)
           + s2_c  * tl.load(w_c + 2 * stride_w_width, mask=mask_ds, other=0.0)
           + new_c * tl.load(w_c + 3 * stride_w_width, mask=mask_ds, other=0.0))
    if HAS_BIAS:
        acc_c = acc_c + tl.load(bias_ptr + c_feat, mask=mask_ds, other=0.0)

    c_out = acc_c * tl.sigmoid(acc_c)  # [BLOCK_DS]  SiLU

    # SSM step  (TIE_HDIM=True: A, dt, dt_bias, D scalar per head)
    dt_val     = tl.load(dt_ptr     + pid_b * stride_dt_batch + pid_h * stride_dt_head).to(tl.float32)
    dt_bias_v  = tl.load(dt_bias_ptr + pid_h).to(tl.float32)
    dt_val     = _softplus(dt_val + dt_bias_v)   # scalar

    a_val = tl.load(A_ptr + pid_h).to(tl.float32)  # negative, pre-exped at load time in vLLM
    dA    = tl.exp(a_val * dt_val)                  # scalar decay

    d_val = tl.load(D_ptr + pid_h).to(tl.float32)  # skip connection, scalar

    # SSM state base: [ssm_slot_in, pid_h, :, :]
    ss_in_base  = ssm_state_ptr + ssm_slot_in  * stride_ss_slot + pid_h * stride_ss_head
    ss_out_base = ssm_state_ptr + ssm_slot_out * stride_ss_slot + pid_h * stride_ss_head

    # Process all (head_dim, dstate) in one tile (BLOCK_HD × BLOCK_DS)
    # Since BLOCK_HD = next_pow2(head_dim) and BLOCK_DS = next_pow2(dstate),
    # this is a single load of the entire SSM state for this head.
    ss_offs = (ss_in_base
               + hd_offs[:, None] * stride_ss_dim
               + ds_offs[None, :] * stride_ss_dstate)
    mask_ss = mask_hd[:, None] & mask_ds[None, :]

    state = tl.load(ss_offs, mask=mask_ss, other=0.0).to(tl.float32)  # [BLOCK_HD, BLOCK_DS]

    # dB = B * dt  (broadcast across head_dim)
    # state update: state = state * dA + dB * x
    dB    = b_out[None, :] * dt_val          # [1, BLOCK_DS]
    state = state * dA + dB * x_out[:, None]  # [BLOCK_HD, BLOCK_DS]

    # Output: y[d] = sum_s(state[d,s] * C[s])
    y = tl.sum(state * c_out[None, :], axis=1)  # [BLOCK_HD]
    y = y + x_out * d_val                       # skip connection

    # Write updated SSM state
    ss_out_offs = (ss_out_base
                   + hd_offs[:, None] * stride_ss_dim
                   + ds_offs[None, :] * stride_ss_dstate)
    tl.store(ss_out_offs, state.to(ssm_state_ptr.dtype.element_ty), mask=mask_ss)

    # Write output
    out_offs = out_ptr + pid_b * stride_out_batch + pid_h * stride_out_head + hd_offs * stride_out_dim
    tl.store(out_offs, y.to(out_ptr.dtype.element_ty), mask=mask_hd)


def mamba2_fused_decode(
    x_bc: torch.Tensor,             # (batch, conv_dim)
    conv_state: torch.Tensor,       # (num_slots, conv_dim, state_len)  — already transposed
    weight: torch.Tensor,           # (conv_dim, kernel_width)
    bias: torch.Tensor | None,      # (conv_dim,)
    conv_state_indices: torch.Tensor,  # (batch,) int32
    ssm_state: torch.Tensor,        # (num_slots, nheads, head_dim, dstate)
    dt: torch.Tensor,               # (batch, nheads)  unexpanded
    A: torch.Tensor,                # (nheads,)
    dt_bias: torch.Tensor,          # (nheads,)
    D: torch.Tensor,                # (nheads,)
    ssm_in_indices: torch.Tensor,   # (batch,) int32
    ssm_out_indices: torch.Tensor,  # (batch,) int32
    out: torch.Tensor,              # (batch, nheads, head_dim)  — pre-allocated, in-place
    n_groups: int,
    pad_slot_id: int = PAD_SLOT_ID,
) -> None:
    """
    Fused Mamba2 decode step.  Replaces:
        hidden_states_B_C = causal_conv1d_update(x_bc, ...)
        selective_state_update(ssm_state, hidden_states, dt, A, B, C, ...)

    Restrictions: KERNEL_WIDTH=4, TIE_HDIM=True, seqlen=1, no APC, no spec-decode.
    Writes results in-place to `out` and `ssm_state`.
    """
    batch       = x_bc.shape[0]
    conv_dim    = x_bc.shape[1]
    _, nheads, head_dim, dstate = ssm_state.shape
    d_inner     = nheads * head_dim
    kernel_width = weight.shape[1]

    assert kernel_width == 4, "mamba2_fused_decode only supports KERNEL_WIDTH=4"
    assert dt.shape == (batch, nheads)
    assert out.shape == (batch, nheads, head_dim)

    BLOCK_HD = triton.next_power_of_2(head_dim)
    BLOCK_DS = triton.next_power_of_2(dstate)

    grid = (batch, nheads)

    with torch.cuda.device(x_bc.device.index):
        _mamba2_fused_decode_kernel[grid](
            x_bc,
            conv_state,
            weight,
            bias,
            conv_state_indices,
            ssm_state,
            dt,
            dt_bias,
            A,
            D,
            ssm_in_indices,
            ssm_out_indices,
            pad_slot_id,
            out,
            # scalar dims
            batch,
            # constexpr dims
            nheads,
            head_dim,
            dstate,
            n_groups,
            d_inner,
            # KERNEL_WIDTH constexpr
            4,
            # strides
            x_bc.stride(0),
            conv_state.stride(0),
            conv_state.stride(1),
            conv_state.stride(2),
            weight.stride(0),
            weight.stride(1),
            ssm_state.stride(0),
            ssm_state.stride(1),
            ssm_state.stride(2),
            ssm_state.stride(3),
            dt.stride(0),
            dt.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            # meta
            bias is not None,
            BLOCK_HD,
            BLOCK_DS,
            num_warps=4,
        )
