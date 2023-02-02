from functools import partial, reduce
from itertools import product
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close, make_tensor

import thunder
import thunder.core.dtypes as datatypes
import thunder.core.lang as tlang
import thunder.langs.torch as ttorch

from .framework import Executor, executors, NOTHING, nvFuser, requiresCUDA, TorchEx


def new_gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class NanoGPTMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        # self.dropout = nn.Dropout()

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        # x = self.dropout(x)
        return x


def thunder_new_gelu(x):
    return 0.5 * x * (1.0 + ttorch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * ttorch.pow(x, 3.0))))


def thunder_NanoGPTMLP_forward_functional(a, c_fc_weight, c_fc_bias, c_proj_weight, c_proj_bias):
    b = ttorch.linear(a, c_fc_weight) + c_fc_bias
    c = thunder_new_gelu(b)
    d = ttorch.linear(c, c_proj_weight, c_proj_bias)
    # e = ttorch.dropout(d)
    return d


# TODO: consider making a deterministic dropout or common source of randomness to enable the dropout
#   comparison
@executors(dtypes=(thunder.float32,))
def test_nanogpt_mlp_functional(executor, device, dtype):
    class Config:
        pass

    n = 4
    config = Config()
    config.n_embd = n
    config.n_head = n
    config.block_size = n

    tdtype = ttorch.torch_dtype(dtype)
    make = partial(make_tensor, dtype=tdtype, device=device)
    mlp = NanoGPTMLP(config).to(device, dtype=tdtype)
    a = make((n, n))

    thunder_fn = thunder.make_traced(thunder_NanoGPTMLP_forward_functional, executor=executor)
    thunder_result = thunder_fn(a, mlp.c_fc.weight, mlp.c_fc.bias, mlp.c_proj.weight, mlp.c_proj.bias)
    torch_result = mlp(a)

    assert_close(thunder_result, torch_result)


class NanoGPTCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # TODO: re-enable me
        # regularization
        # self.attn_dropout = nn.Dropout(config.dropout)
        # self.resid_dropout = nn.Dropout(config.dropout)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # TODO: re-enable me when indexing is supported
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)

        # TODO: re-enable me
        # att = self.attn_dropout(att)

        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)

        # TODO: re-enable me
        # y = self.resid_dropout(y)

        return y


def thunder_NanoGPTCausalSelfAttention_forward_functional(
    x, c_attn_weight, c_attn_bias, n_embd, n_head, bias, c_proj_weight, c_proj_bias
):
    B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
    q, k, v = ttorch.linear(x, c_attn_weight, c_attn_bias).split(n_embd, dim=2)
    k = k.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    q = q.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)
    v = v.view(B, T, n_head, C // n_head).transpose(1, 2)  # (B, nh, T, hs)

    # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

    # att = att.masked_fill(bias[:, :, :T, :T] == 0, float("-inf"))

    att = ttorch.softmax(att, dim=-1)
    y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
    y = ttorch.linear(y, c_proj_weight, c_proj_bias)

    return y


@executors(dtypes=(thunder.float32,))
def test_nanogpt_causalselfattention_functional(executor, device, dtype):
    class Config:
        pass

    n = 4
    config = Config()
    config.n_embd = n
    config.n_head = n
    config.block_size = n

    tdtype = ttorch.torch_dtype(dtype)
    make = partial(make_tensor, dtype=tdtype, device=device)
    csa = NanoGPTCausalSelfAttention(config).to(device, dtype=tdtype)

    a = make((n, n, n))

    torch_result = csa(a)

    thunder_fn = thunder.make_traced(thunder_NanoGPTCausalSelfAttention_forward_functional, executor=executor)
    thunder_result = thunder_fn(
        a,
        csa.c_attn.weight,
        csa.c_attn.bias,
        config.n_embd,
        config.n_head,
        csa.bias,
        csa.c_proj.weight,
        csa.c_proj.bias,
    )

    assert_close(torch_result, thunder_result)


class NanoGPTBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = NanoGPTCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = NanoGPTMLP(config)

    def forward(self, a):
        b = self.ln_1(a)
        a = a + self.attn(b)
        c = self.ln_2(a)
        a = a + self.mlp(c)
        return a


def thunder_NanoGPTBlock_forward_functional(
    a,
    ln_1_normalized_shape,
    ln_1_weight,
    ln_1_bias,
    ln_1_eps,
    csa_c_attn_weight,
    csa_c_attn_bias,
    n_embd,
    n_head,
    csa_bias,
    csa_c_proj_weight,
    csa_c_proj_bias,
    ln_2_normalized_shape,
    ln_2_weight,
    ln_2_bias,
    ln_2_eps,
    c_fc_weight,
    c_fc_bias,
    c_proj_weight,
    c_proj_bias,
):

    b = ttorch.layer_norm(a, ln_1_normalized_shape, ln_1_weight, ln_1_bias, ln_1_eps)
    a = a + thunder_NanoGPTCausalSelfAttention_forward_functional(
        b,
        csa_c_attn_weight,
        csa_c_attn_bias,
        n_embd,
        n_head,
        csa_bias,
        csa_c_proj_weight,
        csa_c_proj_bias,
    )

    c = ttorch.layer_norm(a, ln_2_normalized_shape, ln_2_weight, ln_2_bias, ln_2_eps)
    a = a + thunder_NanoGPTMLP_forward_functional(c, c_fc_weight, c_fc_bias, c_proj_weight, c_proj_bias)

    return a


@executors(dtypes=(thunder.float32,))
def test_nanogpt_block_functional(executor, device, dtype):
    class Config:
        pass

    n = 4
    config = Config()
    config.n_embd = n
    config.n_head = n
    config.block_size = n

    tdtype = ttorch.torch_dtype(dtype)
    make = partial(make_tensor, dtype=tdtype, device=device)
    block = NanoGPTBlock(config).to(device, dtype=tdtype)

    a = make((n, n, n))
    torch_result = block(a)

    thunder_fn = thunder.make_traced(thunder_NanoGPTBlock_forward_functional, executor=executor)
    thunder_result = thunder_fn(
        a,
        block.ln_1.normalized_shape,
        block.ln_1.weight,
        block.ln_1.bias,
        block.ln_1.eps,
        block.attn.c_attn.weight,
        block.attn.c_attn.bias,
        config.n_embd,
        config.n_head,
        block.attn.bias,
        block.attn.c_proj.weight,
        block.attn.c_proj.bias,
        block.ln_2.normalized_shape,
        block.ln_2.weight,
        block.ln_2.bias,
        block.ln_2.eps,
        block.mlp.c_fc.weight,
        block.mlp.c_fc.bias,
        block.mlp.c_proj.weight,
        block.mlp.c_proj.bias,
    )

    assert_close(torch_result, thunder_result)
