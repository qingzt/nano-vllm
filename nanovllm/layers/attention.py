import torch
from torch import nn
from nanovllm.utils.context import get_context
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    HAS_FLASH_ATTN = True
except ImportError:
    import torch.nn.functional as F
    HAS_FLASH_ATTN = False


def _repeat_kv_heads(x: torch.Tensor, num_q_per_kv: int) -> torch.Tensor:
    if num_q_per_kv == 1:
        return x
    return x.unsqueeze(-2).expand(*x.shape[:-2], x.shape[-2], num_q_per_kv, x.shape[-1]).reshape(*x.shape[:-2], x.shape[-2] * num_q_per_kv, x.shape[-1])

if HAS_TRITON:
    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1: return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)

if not HAS_TRITON:
    def store_kvcache_pytorch(
        key: torch.Tensor, 
        value: torch.Tensor, 
        k_cache: torch.Tensor, 
        v_cache: torch.Tensor, 
        slot_mapping: torch.Tensor
    ):
        """
        PyTorch-only replacement for the store_kvcache Triton kernel.
        
        Args:
            key: Shape (num_tokens, num_kv_heads, head_dim)
            value: Shape (num_tokens, num_kv_heads, head_dim)
            k_cache: Shape (num_blocks, block_size, num_kv_heads, head_dim)
            v_cache: Shape (num_blocks, block_size, num_kv_heads, head_dim)
            slot_mapping: Shape (num_tokens,)
        """
        block_size = 256
        block_idx = slot_mapping // block_size
        offset_in_block = slot_mapping % block_size

        k_cache[block_idx, offset_in_block] = key
        v_cache[block_idx, offset_in_block] = value


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    if HAS_TRITON:
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert key.stride(1) == head_dim and value.stride(1) == head_dim
        assert k_cache.stride(1) == D and v_cache.stride(1) == D
        assert slot_mapping.numel() == N
        store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)
    else:
        store_kvcache_pytorch(key, value, k_cache, v_cache, slot_mapping)


if not HAS_FLASH_ATTN:
    def flash_attn_varlen_func(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        max_seqlen_q: int,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_k: int,
        cu_seqlens_k: torch.Tensor,
        softmax_scale: float | None = None,
        causal: bool = False,
        block_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del max_seqlen_q, max_seqlen_k

        output_list = []
        batch_size = len(cu_seqlens_q) - 1
        block_size = k.shape[1] if block_table is not None else None
        num_q_per_kv = q.shape[1] // (k.shape[2] if block_table is not None else k.shape[1])

        for i in range(batch_size):
            q_start, q_end = cu_seqlens_q[i], cu_seqlens_q[i + 1]
            q_seq = q[q_start:q_end]
            seq_len_q = q_seq.shape[0]
            if seq_len_q == 0:
                continue

            if block_table is not None:
                seq_len_k = (cu_seqlens_k[i + 1] - cu_seqlens_k[i]).item()
                if seq_len_k == 0:
                    k_seq = torch.empty(0, q.shape[1], q.shape[2], device=q.device, dtype=q.dtype)
                    v_seq = torch.empty(0, q.shape[1], q.shape[2], device=q.device, dtype=q.dtype)
                else:
                    token_indices = torch.arange(seq_len_k, device=q.device)
                    block_idx_for_token = token_indices // block_size
                    physical_block_nums = block_table[i, block_idx_for_token]
                    offset_in_block = token_indices % block_size
                    k_seq = k[physical_block_nums, offset_in_block]
                    v_seq = v[physical_block_nums, offset_in_block]
            else:
                k_start, k_end = cu_seqlens_k[i], cu_seqlens_k[i + 1]
                k_seq = k[k_start:k_end]
                v_seq = v[k_start:k_end]
                seq_len_k = k_seq.shape[0]

            k_seq_rep = _repeat_kv_heads(k_seq, num_q_per_kv)
            v_seq_rep = _repeat_kv_heads(v_seq, num_q_per_kv)

            q_b = q_seq.permute(1, 0, 2).unsqueeze(0)
            k_b = k_seq_rep.permute(1, 0, 2).unsqueeze(0)
            v_b = v_seq_rep.permute(1, 0, 2).unsqueeze(0)

            attn_mask = None
            if causal and seq_len_k > seq_len_q:
                prefix_len = seq_len_k - seq_len_q
                q_positions = torch.arange(seq_len_q, device=q.device)[:, None]
                k_positions = torch.arange(seq_len_k, device=q.device)[None, :]
                attn_mask = k_positions <= (prefix_len + q_positions)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

            o_seq_b = F.scaled_dot_product_attention(
                q_b,
                k_b,
                v_b,
                attn_mask=attn_mask,
                scale=softmax_scale,
                is_causal=causal and attn_mask is None,
            )
            o_seq = o_seq_b.squeeze(0).permute(1, 0, 2)
            if seq_len_k > seq_len_q:
                o_seq = o_seq[-seq_len_q:]
            output_list.append(o_seq)

        if not output_list:
            return torch.empty_like(q)
        return torch.cat(output_list, dim=0)

    def flash_attn_with_kvcache(
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        cache_seqlens: torch.Tensor,
        block_table: torch.Tensor,
        softmax_scale: float | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        del causal
        batch_size, seq_len_q, num_heads, head_dim = q.shape
        max_seq_len_k = cache_seqlens.max().item()
        block_size = k_cache.shape[1]
        num_q_per_kv = num_heads // k_cache.shape[2]

        token_indices = torch.arange(max_seq_len_k, device=q.device).unsqueeze(0).expand(batch_size, -1)
        block_idx_for_token = token_indices // block_size
        physical_block_nums = block_table.gather(1, block_idx_for_token)
        offset_in_block = token_indices % block_size
        valid_token_mask = token_indices < cache_seqlens.unsqueeze(1)

        k_past = k_cache[physical_block_nums, offset_in_block]
        v_past = v_cache[physical_block_nums, offset_in_block]
        k_past_rep = _repeat_kv_heads(k_past, num_q_per_kv)
        v_past_rep = _repeat_kv_heads(v_past, num_q_per_kv)

        q_b = q.permute(0, 2, 1, 3)
        k_b = k_past_rep.permute(0, 2, 1, 3)
        v_b = v_past_rep.permute(0, 2, 1, 3)
        attn_mask = valid_token_mask.unsqueeze(1).unsqueeze(2)

        o_b = F.scaled_dot_product_attention(
            q_b,
            k_b,
            v_b,
            attn_mask=attn_mask,
            scale=softmax_scale,
            is_causal=False,
        )
        return o_b.permute(0, 2, 1, 3)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        block_size=256, # PagedAttention requires a block size
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size # Store block_size
        
        # GQA repetition factor
        self.num_q_per_kv = self.num_heads // self.num_kv_heads
        
        # Caches are initialized empty
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                    max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                    max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                    softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
            if not HAS_FLASH_ATTN:
                o = o.squeeze(1)
        return o