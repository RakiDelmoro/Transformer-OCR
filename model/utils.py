import copy
import torch
import warnings
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import torch.nn.functional as F
from torch.overrides import has_torch_function, handle_torch_function

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_square_mask(size):
    mask_size = torch.ones(size, size)
    mask = (torch.triu(mask_size)).to(DEVICE).transpose(1, 0)
    mask = mask.float().masked_fill(mask == 0, float("-inf")
                                     ).masked_fill(mask == 1, float(0.0)).to(DEVICE)

    return mask

def stacked_layers(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == 'gelu':
        return F.gelu

    raise RuntimeError(f"activation should either be gelu/relu, not {activation}")

def canonical_mask(mask, mask_name, other_type, other_name,
                   target_type, check_other = True):

    if mask is not None:
        mask_dtype = mask.dtype
        mask_is_float = torch.is_floating_point(mask)
        if mask_dtype != torch.bool and not mask_is_float:
            raise AssertionError(
                f"only bool and floating types of {mask_name} are supported"
            )
        if check_other and other_type is not None:
            if mask_dtype != other_type:
                warnings.warn(
                    f"Support for mismatched {mask_name} and {other_name} " 
                    "is deprecated. Use same type for both instead."
                )
        if not mask_is_float:
            mask = (
                torch.zeros_like(mask, dtype=target_type)
                .masked_fill_(mask, float('-inf'))
            )

    return mask

def none_or_dtype(input_to_check_dtype):
    if input_to_check_dtype is None:
        return None
    elif isinstance(input_to_check_dtype, torch.Tensor):
        return input_to_check_dtype.dtype

    raise RuntimeError("input to none_or_dtype must be None or torch.Tensor")

def scaled_dot_product(q, k, v, head_dim, attn_mask=None):
    scale = head_dim ** -0.5
    attn_scores = torch.matmul(q, k.transpose(-2, -1))  * scale
    if attn_mask is not None:
        attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9).to('cuda')
    attn_probs = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, v)

    return output

def in_projection_packed(q: Tensor, k: Tensor, v: Tensor,
                         w: Tensor, b: Optional[Tensor]=None):

    E = q.shape[-1]
    if k is v:
        if q is k:
            proj = F.linear(q, w, b)
            # reshape to 3, E and not E, 3 is delibrate for better memory coalescing and keeping same order as chunk
            proj = proj.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
            return proj[0], proj[1], proj[2]
        else:
            # encoder-decoder attention
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            q_proj = F.linear(q, w_q, b_q)
            kv_proj = F.linear(k, w_kv, b_kv)
            # reshape to 2, E and not E, 2 is delibrate for better memory coalescing and keeping same order as chunk
            kv_proj = kv_proj.unflatten(-1, (2, E)).unsqueeze(0).transpose(0, -2).unsqueeze(-2).contiguous()
            return (q_proj, kv_proj[0], kv_proj[1])
    else:
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

def multi_head_attention_forward(query: Tensor, key: Tensor, value: Tensor,
                                 num_heads: int, in_proj_weight: Optional[Tensor],
                                 in_proj_bias: Optional[Tensor],
                                 out_proj_weight: Optional[Tensor],
                                 out_proj_bias: Optional[Tensor],
                                 key_padding_mask: Optional[Tensor]=None,
                                 attn_mask: Optional[Tensor]=None):

    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias)

    if has_torch_function(tens_ops):
        return handle_torch_function(multi_head_attention_forward,
                                     tens_ops,
                                     query,
                                     key,
                                     value,
                                     num_heads,
                                     in_proj_weight,
                                     in_proj_bias,
                                     out_proj_weight,
                                     out_proj_bias,
                                     attn_mask,
                                     key_padding_mask)

    # set up shape variables
    trgt_len, batch, embed_dim = query.shape
    src_len, _, _, = key.shape
    head_dim = embed_dim // num_heads
    
    key_padding_mask = canonical_mask(
    mask=key_padding_mask,
    mask_name="key_padding_mask",
    other_type=none_or_dtype(attn_mask),
    other_name="attn_mask",
    target_type=query.dtype)

    attn_mask = canonical_mask(
        mask=attn_mask,
        mask_name="attn_mask",
        other_type=None,
        other_name="",
        target_type=query.dtype,
        check_other=False)

    q, k, v = in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)

    q = q.view(trgt_len, batch*num_heads, head_dim).transpose(0, 1)
    k = k.view(key.shape[0], batch*num_heads, head_dim).transpose(0, 1)
    v = v.view(value.shape[0], batch*num_heads, head_dim).transpose(0, 1)
    
    src_len = k.shape[1]

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (batch, src_len), \
            f"expecting key_padding_mask of {(batch, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(batch, 1, 1, src_len). \
            expand(-1, num_heads, -1, -1).reshape(batch * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    if attn_mask is not None:
        if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(0)
        else:
            attn_mask = attn_mask.view(batch, num_heads, -1, src_len)

    q = q.view(batch, num_heads, trgt_len, head_dim)
    k = k.view(batch, num_heads, src_len, head_dim)
    v = v.view(batch, num_heads, src_len, head_dim)

    attn_output = scaled_dot_product(q, k, v, head_dim, attn_mask)
    attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(batch*trgt_len, embed_dim)

    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(trgt_len, batch, embed_dim)

    return attn_output