# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.import functools

import copy
import functools
import json
import os
import time
from collections import defaultdict
from typing import List, Optional

import numpy as np
import safetensors
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D
from pathlib import Path

from ..._utils import pad_vocab_size, str_dtype_to_torch
from ...logger import logger
from ...mapping import Mapping
from ...quantization import QuantAlgo
from ..convert_utils import (dup_kv_weight, generate_int8, get_weight,
                             get_weight_and_bias, load_calib_dataset,
                             smooth_gemm, smooth_gemm_fc1_gate, split,
                             split_matrix_tp, split_qkv_bias_tp, split_qkv_tp)
from .config import QWenConfig
from .utils import get_qwen_key_list, make_context

from ...quantization.quantize import (qserve_pack_reorder_per_channel,
                                      qserve_pack_reorder_per_group,
                                      qserve_quantize_weight_per_channel,
                                      qserve_quantize_weight_per_group)


@torch.no_grad()
def smooth_qwen_model(model, scales, alpha, qwen_qkv_para, qwen_smoother):
    # Smooth the activation and weights with smoother = $\diag{s}$
    for name, module in model.named_modules():
        if not module._get_name() == "QWenBlock":
            continue
        # qkv_proj
        layer_name = name + ".attn.c_attn"
        smoother = smooth_gemm(module.attn.c_attn.weight,
                               scales[layer_name]["x"], module.ln_1.weight,
                               None, alpha)

        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.attn.c_attn.weight.abs().max(dim=1)[0]

        # see transpose_weights function
        qwen_qkv_para[layer_name] = module.attn.c_attn.weight.transpose(
            0, 1).contiguous()

        # =================================================================
        layer_name = name + ".attn.c_proj"
        smoother = smooth_gemm(
            module.attn.c_proj.weight,
            scales[layer_name]["x"],
            None,
            None,
            alpha=alpha,
        )
        qwen_smoother[layer_name] = smoother.float()

        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.attn.c_proj.weight.abs().max(dim=1)[0]
        # ==================================================================
        fc1_layer_name = name + ".mlp.w1"
        gate_layer_name = name + ".mlp.w2"

        smoother = smooth_gemm_fc1_gate(module.mlp.w1.weight,
                                        module.mlp.w2.weight,
                                        scales[fc1_layer_name]["x"],
                                        module.ln_2.weight, None, alpha)

        scales[fc1_layer_name]["x"] = scales[fc1_layer_name]["x"] / smoother
        scales[fc1_layer_name]["w"] = module.mlp.w1.weight.abs().max(dim=1)[0]

        scales[gate_layer_name]["x"] = scales[gate_layer_name]["x"] / smoother
        scales[gate_layer_name]["w"] = module.mlp.w2.weight.abs().max(dim=1)[0]

        # ==================================================================
        layer_name = name + ".mlp.c_proj"
        smoother = smooth_gemm(module.mlp.c_proj.weight,
                               scales[layer_name]["x"], None, None, alpha)
        qwen_smoother[layer_name] = smoother.float()
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.mlp.c_proj.weight.abs().max(dim=1)[0]


@torch.no_grad()
def smooth_qwen2_model(model, scales, alpha, qwen_qkv_para, qwen_smoother):
    # Smooth the activation and weights with smoother = $\diag{s}$
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    for name, module in model.named_modules():
        if not isinstance(module, Qwen2DecoderLayer):
            continue
        # qkv_proj
        layer_name_q = name + ".self_attn.q_proj"
        layer_name_k = name + ".self_attn.k_proj"
        layer_name_v = name + ".self_attn.v_proj"
        layer_name_qkv = name + ".self_attn.qkv_proj"

        weight = torch.cat([
            module.self_attn.q_proj.weight, module.self_attn.k_proj.weight,
            module.self_attn.v_proj.weight
        ],
                           dim=0)

        smoother = smooth_gemm(weight, scales[layer_name_q]["x"],
                               module.input_layernorm.weight, None, alpha)

        scales[layer_name_qkv]["x"] = scales[layer_name_q]["x"] / smoother
        scales[layer_name_qkv]["w"] = weight.abs().max(dim=1)[0]
        scales[layer_name_qkv]["y"] = torch.cat([
            scales[layer_name_q]["y"], scales[layer_name_k]["y"],
            scales[layer_name_v]["y"]
        ],
                                                dim=0)

        # see transpose_weights function
        qwen_qkv_para[layer_name_qkv] = weight.transpose(0, 1).contiguous()

        # =================================================================
        layer_name = name + ".self_attn.o_proj"
        smoother = smooth_gemm(module.self_attn.o_proj.weight,
                               scales[layer_name]["x"], None, None, alpha)
        qwen_smoother[layer_name] = smoother.float()

        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.self_attn.o_proj.weight.abs().max(
            dim=1)[0]

        # ==================================================================
        fc1_layer_name = name + ".mlp.gate_proj"
        gate_layer_name = name + ".mlp.up_proj"

        smoother = smooth_gemm_fc1_gate(module.mlp.gate_proj.weight,
                                        module.mlp.up_proj.weight,
                                        scales[fc1_layer_name]["x"],
                                        module.post_attention_layernorm.weight,
                                        None, alpha)

        scales[fc1_layer_name]["x"] = scales[fc1_layer_name]["x"] / smoother
        scales[fc1_layer_name]["w"] = module.mlp.gate_proj.weight.abs().max(
            dim=1)[0]

        scales[gate_layer_name]["x"] = scales[gate_layer_name]["x"] / smoother
        scales[gate_layer_name]["w"] = module.mlp.up_proj.weight.abs().max(
            dim=1)[0]

        # ==================================================================
        layer_name = name + ".mlp.down_proj"
        smoother = smooth_gemm(module.mlp.down_proj.weight,
                               scales[layer_name]["x"], None, None, alpha)
        qwen_smoother[layer_name] = smoother.float()
        scales[layer_name]["x"] = scales[layer_name]["x"] / smoother
        scales[layer_name]["w"] = module.mlp.down_proj.weight.abs().max(
            dim=1)[0]


@torch.no_grad()
def capture_activation_range(model,
                             qwen_type,
                             tokenizer,
                             dataset,
                             system_prompt,
                             chat_format,
                             num_samples=512,
                             seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    act_scales = defaultdict(lambda: {"x": None, "y": None, "w": None})

    if qwen_type == 'qwen':
        tokenizer.pad_token_id = tokenizer.im_end_id
    else:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def stat_tensor(name, tensor, act_scales, key):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_max = torch.max(tensor, dim=0)[0].float()

        if act_scales[name][key] is None:
            act_scales[name][key] = comming_max
        else:
            act_scales[name][key] = torch.max(act_scales[name][key],
                                              comming_max)

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x, act_scales, "x")
        stat_tensor(name, y, act_scales, "y")

        if act_scales[name]["w"] is None:
            act_scales[name]["w"] = m.weight.abs().clip(1e-8,
                                                        None).max(dim=1)[0]

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) or isinstance(m, Conv1D):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(num_samples), desc="calibrating model"):
        line = dataset[i]
        line = line + ' TL;DR: '
        line = line.strip()
        line = line.replace(" n't", "n't")
        if qwen_type == 'qwen':
            _, input_id_list = make_context(tokenizer=tokenizer,
                                            query=line,
                                            history=[],
                                            system=system_prompt,
                                            chat_format=chat_format,
                                            max_input_length=seq_len)
            line_encoded = torch.from_numpy(
                np.array(input_id_list,
                         dtype=np.int32)).type(torch.int32).unsqueeze(0)
            line_encoded = line_encoded.to(device)
        else:
            line_encoded = tokenizer(line,
                                     return_tensors="pt",
                                     max_length=seq_len,
                                     padding=True,
                                     truncation=True).input_ids.to(device)
        model(line_encoded)
    for h in hooks:
        h.remove()
    return act_scales


def get_tllm_linear_weight(weight,
                           prefix,
                           bias=None,
                           use_weight_only=False,
                           plugin_weight_only_quant_type=torch.int8,
                           dtype='float32',
                           use_gemm_woq_plugin=True,
                           postfix='weight',
                           quant_scale_name=None):
    results = {}
    if use_weight_only:
        if weight.dim() > 2:
            v = weight.transpose(1, 2).contiguous().clone()
        else:
            v = weight.t().contiguous().clone()
        processed_torch_weights, torch_weight_scales = \
            torch.ops.trtllm.symmetric_quantize_last_axis_of_batched_matrix(
                v.cpu(), plugin_weight_only_quant_type)
        if not use_gemm_woq_plugin:
            results[prefix + postfix] = v.to(dtype)
        else:
            results[prefix + postfix] = processed_torch_weights
        if quant_scale_name is not None:
            results[quant_scale_name] = torch_weight_scales
        else:
            results[prefix + 'per_channel_scale'] = torch_weight_scales
    else:
        results[prefix + postfix] = weight.clone()

    if bias is not None:
        results[prefix + 'bias'] = bias

    return results


def get_tllm_linear_sq_weight(vals,
                              prefix,
                              shape,
                              tensor_parallel,
                              is_qkv=False,
                              per_token=False,
                              per_channel=False,
                              last_prefix=None,
                              bias=None,
                              smoother_value=None,
                              smoother_shape=None,
                              rank=0,
                              cat_dim=0,
                              multi_query_mode=False):
    results = {}

    def multi_query_split(data, local_dim, head_size, tp_size, cur_rank):

        q, k, v = torch.split(data, [local_dim, head_size, head_size], dim=-1)
        q_split = torch.split(q, q.shape[-1] // tp_size, dim=-1)
        k_split = torch.split(k, k.shape[-1] // tp_size, dim=-1)
        v_split = torch.split(v, v.shape[-1] // tp_size, dim=-1)
        return [
            torch.concat((q_split[ii], k_split[ii], v_split[ii]), dim=-1)
            for ii in range(tp_size)
        ][cur_rank]

    col_shape = shape if (is_qkv or per_channel) else [1, 1]

    if per_token:
        if per_channel:
            original_weights = vals["weight.int8.col"]
        else:
            original_weights = vals["weight.int8"]
        local_dim = original_weights.shape[0]
        head_size = (original_weights.shape[1] - local_dim) // 2

        if multi_query_mode:
            cur_weights = multi_query_split(original_weights, local_dim,
                                            head_size, tensor_parallel, rank)
        else:
            cur_weights = torch.chunk(original_weights,
                                      tensor_parallel,
                                      dim=cat_dim)[rank]
        if is_qkv:
            hidden_dim = cur_weights.shape[0]
            cur_weights = cur_weights.reshape(hidden_dim, -1)
        results[prefix + 'weight'] = cur_weights.t().contiguous()
        if smoother_value is None:
            results[last_prefix] = torch.from_numpy(
                np.array([1.0], dtype=np.float32))

        if per_channel:
            cur_per_channel_value = vals["scale_w_quant_orig.col"]
            if smoother_value is None:
                if multi_query_mode:

                    cur_per_channel_value = multi_query_split(
                        vals["scale_w_quant_orig.col"], local_dim, head_size,
                        tensor_parallel, rank)
                else:
                    cur_per_channel_value = np.split(
                        vals["scale_w_quant_orig.col"],
                        tensor_parallel,
                        axis=cat_dim)[rank]
        else:
            cur_per_channel_value = vals["scale_w_quant_orig"]
            if is_qkv:
                if multi_query_mode:
                    cur_per_channel_value = multi_query_split(
                        vals["scale_w_quant_orig"], local_dim, head_size,
                        tensor_parallel, rank)
                else:
                    cur_per_channel_value = torch.split(
                        vals["scale_w_quant_orig"],
                        tensor_parallel,
                        axis=cat_dim)[rank]

        results[prefix +
                'per_channel_scale'] = cur_per_channel_value.reshape(col_shape)
    else:
        if per_channel:
            original_weights = vals["weight.int8.col"]
        else:
            original_weights = vals["weight.int8"]
        local_dim = original_weights.shape[0]
        head_size = (original_weights.shape[1] - local_dim) // 2

        if multi_query_mode:
            cur_weights = multi_query_split(original_weights, local_dim,
                                            head_size, tensor_parallel, rank)
        else:
            cur_weights = torch.chunk(original_weights,
                                      tensor_parallel,
                                      dim=cat_dim)[rank]
        if is_qkv:
            hidden_dim = cur_weights.shape[0]
            cur_weights = cur_weights.reshape(hidden_dim, -1)
        results[prefix + 'weight'] = cur_weights.t().contiguous()

        if per_channel:
            cur_per_channel_value = vals["scale_y_accum_quant.col"]
            if smoother_value is None:
                if multi_query_mode:
                    cur_per_channel_value = multi_query_split(
                        vals["scale_y_accum_quant.col"], local_dim, head_size,
                        tensor_parallel, rank)
                else:
                    cur_per_channel_value = np.split(
                        vals["scale_y_accum_quant.col"],
                        tensor_parallel,
                        axis=cat_dim)[rank]
        else:
            cur_per_channel_value = vals["scale_y_accum_quant"]
            # QKV is always per_channel
            if is_qkv:
                if multi_query_mode:
                    cur_per_channel_value = multi_query_split(
                        vals["scale_y_accum_quant"], local_dim, head_size,
                        tensor_parallel, rank)
                else:
                    cur_per_channel_value = np.split(
                        vals["scale_y_accum_quant"],
                        tensor_parallel,
                        axis=cat_dim)[rank]

        results[prefix + 'per_channel_scale'] = cur_per_channel_value.reshape(
            col_shape).contiguous()

        results[last_prefix] = vals['scale_x_orig_quant'].contiguous()

        results[prefix + 'act_scale'] = vals["scale_y_quant_orig"].contiguous()

    if smoother_value is not None:
        cur_smoother_value = torch.split(smoother_value,
                                         smoother_value.shape[-1] //
                                         tensor_parallel,
                                         dim=cat_dim)[rank]

        results[prefix + 'smoother'] = cur_smoother_value.reshape(
            smoother_shape).contiguous().to(torch.float32)

    if bias is not None:
        results[prefix + 'bias'] = bias

    return results


def load_hf_qwen(model_dir: str, load_model_on_cpu: bool = False):
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    if config['architectures'] == ['Qwen2ForSequenceClassification']:
        from transformers import Qwen2ForSequenceClassification as model_cls
    else:
        from transformers import AutoModelForCausalLM as model_cls

    model = model_cls.from_pretrained(
        model_dir,
        device_map='auto' if not load_model_on_cpu else 'cpu',
        torch_dtype='auto',
        trust_remote_code=True)
    return model


def convert_hf_qwen(hf_model,
                    qwen_type,
                    mapping: Mapping,
                    vocab_size=32000,
                    dtype='float32',
                    use_parallel_embedding=False,
                    sharding_dim=0,
                    use_weight_only=False,
                    share_embedding_table=False,
                    use_gemm_woq_plugin=False,
                    plugin_weight_only_quant_type=torch.int8,
                    use_smooth_quant=False,
                    per_channel=False,
                    per_token=False,
                    int8_kv_cache=False,
                    act_range=[],
                    qkv_para=[],
                    smoother=[],
                    moe_config=None):
    weights = {}
    tik = time.time()
    tensor_parallel = mapping.tp_size
    model_params = dict(hf_model.named_parameters())

    dtype = getattr(torch, dtype)
    hf_config = hf_model.config
    if hasattr(hf_config, 'llm_config'):
        hf_config = hf_config.llm_config

    #This is for InternVL2 - 1B
    keys_to_rename = [
        key for key in model_params.keys() if 'language_model.' in key
    ]
    keys_to_delete = [
        key for key in model_params.keys() if 'vision_model.' in key
    ]
    for key in keys_to_rename:
        keys_rename = key.replace('language_model.', '')
        model_params[keys_rename] = model_params[key]
        del model_params[key]
    for key in keys_to_delete:
        del model_params[key]

    num_attention_heads = hf_config.num_attention_heads
    hidden_size = hf_config.hidden_size
    head_size = hidden_size // num_attention_heads
    if qwen_type == 'qwen':
        intermediate_size = hf_config.intermediate_size // 2  # Qwen version 1 has actual intermediate_size one half of what's in hf_config
    else:
        intermediate_size = hf_config.intermediate_size
    num_key_value_heads = hf_config.num_key_value_heads if hasattr(
        hf_config, "num_key_value_heads") else num_attention_heads
    mha_mode = (num_key_value_heads == num_attention_heads)
    layers_range = mapping.pp_layers(hf_config.num_hidden_layers)

    layer_prefix = "transformer.h." if qwen_type == 'qwen' else "model.layers."
    key_list = get_qwen_key_list(qwen_type)

    for l in layers_range:
        prefix = layer_prefix + f'{l}.'
        tllm_prex = f'transformer.layers.{l - layers_range[0]}.'
        if qwen_type == 'qwen':
            qkv_weight, qkv_bias = get_weight_and_bias(model_params,
                                                       prefix + key_list[0],
                                                       dtype)
            qkv_w = split_qkv_tp(qkv_weight, num_attention_heads, hidden_size,
                                 tensor_parallel, mapping.tp_rank)
            qkv_b = split_qkv_bias_tp(qkv_bias, num_attention_heads,
                                      hidden_size, tensor_parallel,
                                      mapping.tp_rank)
        else:
            q_weight, q_bias = get_weight_and_bias(
                model_params, prefix + key_list[0] + 'q_proj', dtype)
            k_weight, k_bias = get_weight_and_bias(
                model_params, prefix + key_list[0] + 'k_proj', dtype)
            v_weight, v_bias = get_weight_and_bias(
                model_params, prefix + key_list[0] + 'v_proj', dtype)
            if not mha_mode:
                if num_key_value_heads < tensor_parallel:
                    # duplicate the KV heads up to tensor_parallel
                    k_weight = dup_kv_weight(k_weight, num_key_value_heads,
                                             tensor_parallel)
                    v_weight = dup_kv_weight(v_weight, num_key_value_heads,
                                             tensor_parallel)
                    k_bias = dup_kv_weight(k_bias, num_key_value_heads,
                                           tensor_parallel)
                    v_bias = dup_kv_weight(v_bias, num_key_value_heads,
                                           tensor_parallel)
                assert (k_weight.shape[0] % (mapping.tp_size * head_size)) == 0
                assert (v_weight.shape[0] % (mapping.tp_size * head_size)) == 0
                assert (k_bias.shape[0] % (mapping.tp_size * head_size)) == 0
                assert (v_bias.shape[0] % (mapping.tp_size * head_size)) == 0

                wq = split(q_weight, mapping.tp_size, mapping.tp_rank)
                wk = split(k_weight, mapping.tp_size, mapping.tp_rank)
                wv = split(v_weight, mapping.tp_size, mapping.tp_rank)

                bq = split(q_bias, mapping.tp_size, mapping.tp_rank)
                bk = split(k_bias, mapping.tp_size, mapping.tp_rank)
                bv = split(v_bias, mapping.tp_size, mapping.tp_rank)

                qkv_w = torch.concat((wq, wk, wv))
                qkv_b = torch.concat((bq, bk, bv))
            else:
                qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
                qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)

                qkv_w = split_qkv_tp(qkv_weight, num_attention_heads,
                                     hidden_size, tensor_parallel,
                                     mapping.tp_rank)
                qkv_b = split_qkv_bias_tp(qkv_bias, num_attention_heads,
                                          hidden_size, tensor_parallel,
                                          mapping.tp_rank)

        if use_smooth_quant:
            qkv_proj_key = key_list[
                0] if qwen_type == 'qwen' else 'self_attn.qkv_proj'
            qkv_weight = qkv_para[prefix + qkv_proj_key]
            qkv_out_dim = qkv_weight.shape[1]

            if not mha_mode:
                local_dim = qkv_weight.shape[0]
                kv_hidden_size = (qkv_weight.shape[-1] - local_dim) // 2
                qkv_weight = qkv_weight.reshape(local_dim,
                                                local_dim + 2 * kv_hidden_size)
            else:
                qkv_weight = qkv_weight.reshape(hidden_size, 3, hidden_size)

            int8_weights = generate_int8(qkv_weight,
                                         act_range.get(prefix + qkv_proj_key),
                                         is_qkv=True,
                                         multi_query_mode=bool(not mha_mode))

            weights.update(
                get_tllm_linear_sq_weight(int8_weights,
                                          tllm_prex + 'attention.qkv.',
                                          [1, qkv_out_dim // tensor_parallel],
                                          tensor_parallel,
                                          is_qkv=True,
                                          per_token=per_token,
                                          per_channel=per_channel,
                                          last_prefix=tllm_prex +
                                          'input_layernorm.scale_to_int',
                                          bias=qkv_b,
                                          smoother_value=None,
                                          smoother_shape=None,
                                          rank=mapping.tp_rank,
                                          cat_dim=-1,
                                          multi_query_mode=bool(not mha_mode)))
        else:
            weights.update(
                get_tllm_linear_weight(qkv_w, tllm_prex + 'attention.qkv.',
                                       qkv_b, use_weight_only,
                                       plugin_weight_only_quant_type, dtype,
                                       use_gemm_woq_plugin))
        if int8_kv_cache:
            if qwen_type == 'qwen':
                qkv_y = act_range.get(prefix + key_list[0])["y"]
            else:
                qkv_y = torch.cat([
                    act_range.get(prefix + key_list[0] + 'q_proj')["y"],
                    act_range.get(prefix + key_list[0] + 'k_proj')["y"],
                    act_range.get(prefix + key_list[0] + 'v_proj')["y"]
                ],
                                  dim=0)

            int8_kv_scales = qkv_y.max() / 127.

            kv_cache_weights = {}

            kv_cache_weights[
                tllm_prex +
                'attention.kv_cache_scaling_factor'] = int8_kv_scales.reshape(
                    [1])

            weights.update(kv_cache_weights)

        attn_dense_weight = get_weight(model_params, prefix + key_list[1],
                                       dtype)
        split_v = split_matrix_tp(attn_dense_weight,
                                  tensor_parallel,
                                  mapping.tp_rank,
                                  dim=1)
        if use_smooth_quant:
            attn_dense_weight = attn_dense_weight.t()
            int8_weights = generate_int8(attn_dense_weight,
                                         act_range.get(prefix + key_list[1]))
            weights.update(
                get_tllm_linear_sq_weight(
                    int8_weights,
                    tllm_prex + 'attention.dense.', [1, hidden_size],
                    tensor_parallel,
                    is_qkv=False,
                    per_token=per_token,
                    per_channel=per_channel,
                    last_prefix=tllm_prex +
                    'attention.quantization_scaling_factor',
                    smoother_value=smoother[(prefix + key_list[1])],
                    smoother_shape=[1, hidden_size // tensor_parallel],
                    rank=mapping.tp_rank,
                    cat_dim=0))
        else:
            weights.update(
                get_tllm_linear_weight(split_v, tllm_prex + 'attention.dense.',
                                       None, use_weight_only,
                                       plugin_weight_only_quant_type, dtype,
                                       use_gemm_woq_plugin))

        if qwen_type == "qwen2_moe" and moe_config and moe_config.has_moe():

            # shared_expert for qwen2_moe
            shared_expert_up_proj = model_params[
                f'model.layers.{l}.mlp.shared_expert.up_proj.weight']
            shared_expert_down_proj = model_params[
                f'model.layers.{l}.mlp.shared_expert.down_proj.weight']
            shared_expert_gate = model_params[
                f'model.layers.{l}.mlp.shared_expert.gate_proj.weight']
            shared_expert_up_proj = split(shared_expert_up_proj,
                                          mapping.tp_size,
                                          mapping.tp_rank,
                                          dim=0)
            shared_expert_down_proj = split(shared_expert_down_proj,
                                            mapping.tp_size,
                                            mapping.tp_rank,
                                            dim=1)
            shared_expert_gate = split(shared_expert_gate,
                                       mapping.tp_size,
                                       mapping.tp_rank,
                                       dim=0)
            shared_expert_gate_up_proj = torch.concat(
                [shared_expert_up_proj, shared_expert_gate], dim=-2).to(dtype)

            ## mlp.shared_expert.gate_up_proj.weight
            weights.update(
                get_tllm_linear_weight(shared_expert_gate_up_proj,
                                       tllm_prex + 'shared_expert.fc.', None,
                                       use_weight_only,
                                       plugin_weight_only_quant_type, dtype,
                                       use_gemm_woq_plugin))

            ## mlp.shared_expert.down_proj.weight
            weights.update(
                get_tllm_linear_weight(shared_expert_down_proj.to(dtype),
                                       tllm_prex + 'shared_expert.proj.', None,
                                       use_weight_only,
                                       plugin_weight_only_quant_type, dtype,
                                       use_gemm_woq_plugin))

            moe_shared_expert_gate_weights = get_weight(
                model_params, prefix + 'mlp.shared_expert_gate', dtype)
            weights.update(
                get_tllm_linear_weight(
                    moe_shared_expert_gate_weights,
                    tllm_prex + 'shared_expert_gate.',
                    None,
                    False,  # Router should never be quantized
                    plugin_weight_only_quant_type,
                    dtype,
                    use_gemm_woq_plugin))

            ## fine-grained experts
            rank_experts = list(range(moe_config.num_experts))
            if mapping.has_moe_ep():
                rank_experts = mapping.ep_experts(moe_config.num_experts)
            for suffix in ["gate_proj", "down_proj", "up_proj"]:
                model_params[f'model.layers.{l}.mlp.experts.{suffix}.weight'] = \
                            torch.stack([model_params[f'model.layers.{l}.mlp.experts.{expert}.{suffix}.weight'].detach()
                                        for expert in rank_experts])
            w3 = model_params[f'model.layers.{l}.mlp.experts.up_proj.weight']
            w2 = model_params[f'model.layers.{l}.mlp.experts.down_proj.weight']
            w1 = model_params[f'model.layers.{l}.mlp.experts.gate_proj.weight']
            if mapping.has_moe_tp():
                w3 = split(w3, mapping.moe_tp_size, mapping.moe_tp_rank, dim=1)
                w2 = split(w2, mapping.moe_tp_size, mapping.moe_tp_rank, dim=2)
                w1 = split(w1, mapping.moe_tp_size, mapping.moe_tp_rank, dim=1)

            moe_experts_w3w1_weights = torch.concat([w3, w1], dim=-2).to(dtype)

            ## mlp.experts.w2.weight
            weights.update(
                get_tllm_linear_weight(w2.to(dtype), tllm_prex + 'mlp.proj.',
                                       None, use_weight_only,
                                       plugin_weight_only_quant_type, dtype,
                                       use_gemm_woq_plugin))

            ## mlp.experts.w3w1.weight
            weights.update(
                get_tllm_linear_weight(moe_experts_w3w1_weights,
                                       tllm_prex + 'mlp.fc.', None,
                                       use_weight_only,
                                       plugin_weight_only_quant_type, dtype,
                                       use_gemm_woq_plugin))

            moe_experts_gate_weights = get_weight(model_params,
                                                  prefix + 'mlp.gate',
                                                  torch.float32)
            weights.update(
                get_tllm_linear_weight(
                    moe_experts_gate_weights,
                    tllm_prex + 'mlp.router.',
                    None,
                    False,  # Router should never be quantized
                    plugin_weight_only_quant_type,
                    dtype,
                    use_gemm_woq_plugin))
        else:
            mlp_gate_weight = get_weight(model_params, prefix + key_list[2],
                                         dtype)
            split_v = split_matrix_tp(mlp_gate_weight,
                                      tensor_parallel,
                                      mapping.tp_rank,
                                      dim=0)
            if use_smooth_quant:
                mlp_gate_weight = mlp_gate_weight.t()
                int8_weights = generate_int8(
                    mlp_gate_weight, act_range.get(prefix + key_list[2]))

                weights.update(
                    get_tllm_linear_sq_weight(
                        int8_weights,
                        tllm_prex + 'mlp.gate.',
                        [1, intermediate_size // tensor_parallel],
                        tensor_parallel,
                        is_qkv=False,
                        per_token=per_token,
                        per_channel=per_channel,
                        last_prefix=tllm_prex + 'post_layernorm.scale_to_int',
                        smoother_value=None,
                        smoother_shape=None,
                        rank=mapping.tp_rank,
                        cat_dim=-1))
            else:
                weights.update(
                    get_tllm_linear_weight(split_v, tllm_prex + 'mlp.gate.',
                                           None, use_weight_only,
                                           plugin_weight_only_quant_type, dtype,
                                           use_gemm_woq_plugin))

            mlp_fc_weight = get_weight(model_params, prefix + key_list[3],
                                       dtype)
            split_v = split_matrix_tp(mlp_fc_weight,
                                      tensor_parallel,
                                      mapping.tp_rank,
                                      dim=0)

            if use_smooth_quant:
                mlp_fc_weight = mlp_fc_weight.t()  #verified
                int8_weights = generate_int8(
                    mlp_fc_weight, act_range.get(prefix + key_list[3]))

                weights.update(
                    get_tllm_linear_sq_weight(
                        int8_weights,
                        tllm_prex + 'mlp.fc.',
                        [1, intermediate_size // tensor_parallel],
                        tensor_parallel,
                        is_qkv=False,
                        per_token=per_token,
                        per_channel=per_channel,
                        last_prefix=tllm_prex + 'post_layernorm.scale_to_int',
                        smoother_value=None,
                        smoother_shape=None,
                        rank=mapping.tp_rank,
                        cat_dim=-1))
            else:
                weights.update(
                    get_tllm_linear_weight(split_v, tllm_prex + 'mlp.fc.', None,
                                           use_weight_only,
                                           plugin_weight_only_quant_type, dtype,
                                           use_gemm_woq_plugin))

            mlp_proj_weight = get_weight(model_params, prefix + key_list[4],
                                         dtype)
            split_v = split_matrix_tp(mlp_proj_weight,
                                      tensor_parallel,
                                      mapping.tp_rank,
                                      dim=1)

            if use_smooth_quant:
                mlp_proj_weight = mlp_proj_weight.t()
                int8_weights = generate_int8(
                    mlp_proj_weight, act_range.get(prefix + key_list[4]))

                weights.update(
                    get_tllm_linear_sq_weight(
                        int8_weights,
                        tllm_prex + 'mlp.proj.', [1, hidden_size],
                        tensor_parallel,
                        is_qkv=False,
                        per_token=per_token,
                        per_channel=per_channel,
                        last_prefix=tllm_prex +
                        'mlp.quantization_scaling_factor',
                        smoother_value=smoother[prefix + key_list[4]],
                        smoother_shape=[
                            1, intermediate_size // tensor_parallel
                        ],
                        rank=mapping.tp_rank,
                        cat_dim=0))
            else:
                weights.update(
                    get_tllm_linear_weight(split_v, tllm_prex + 'mlp.proj.',
                                           None, use_weight_only,
                                           plugin_weight_only_quant_type, dtype,
                                           use_gemm_woq_plugin))

        # Layer norms do not use tensor parallelism
        input_ln_weight = get_weight(model_params, prefix + key_list[5], dtype)
        weights[tllm_prex + 'input_layernorm.weight'] = input_ln_weight

        post_ln_weight = get_weight(model_params, prefix + key_list[6], dtype)
        weights[tllm_prex + 'post_layernorm.weight'] = post_ln_weight

    v = get_weight(model_params, key_list[7], dtype)

    if mapping.is_last_pp_rank():
        if hf_config.tie_word_embeddings:
            # lm_head.weight has the same weights as embedding
            lm_head_weights = v.clone()
        else:
            lm_head_weights = get_weight(model_params, 'lm_head', dtype)

        if vocab_size % mapping.tp_size != 0:
            # padding
            vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
            pad_width = vocab_size_padded - vocab_size

            lm_head_weights = torch.from_numpy(
                np.pad(lm_head_weights.detach().cpu().numpy(),
                       ((0, pad_width), (0, 0)),
                       'constant',
                       constant_values=0))
        weights['lm_head.weight'] = split_matrix_tp(lm_head_weights,
                                                    tensor_parallel,
                                                    mapping.tp_rank,
                                                    dim=0)

    if use_parallel_embedding:
        v = split_matrix_tp(v,
                            mapping.tp_size,
                            mapping.tp_rank,
                            dim=sharding_dim)

    if mapping.is_first_pp_rank():
        weights['transformer.vocab_embedding.weight'] = v

    if mapping.is_last_pp_rank():
        ln_f_w = get_weight(model_params, key_list[8], dtype)
        weights['transformer.ln_f.weight'] = ln_f_w

    if hasattr(hf_model, 'score'):
        score = get_weight(model_params, 'score', dtype)
        weights['lm_head.weight'] = score

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Weights loaded. Total time: {t}')
    return weights


def quantize(hf_model_dir: str,
             output_dir: str,
             config: QWenConfig,
             calib_dataset='cnn_dailymail'):
    '''
        Quantize the save the model as TRT-LLM checkpoint to output_dir
    '''
    os.makedirs(output_dir, exist_ok=True)
    config.to_json_file(os.path.join(output_dir, 'config.json'))

    mapping = config.mapping
    assert mapping.rank == 0, "quantize should be called at rank 0 only"

    quant_config = config.quantization
    use_smooth_quant = quant_config.use_plugin_sq
    int8_kv_cache = quant_config.kv_cache_quant_algo == "INT8"

    assert use_smooth_quant or int8_kv_cache, "Call from_hugging_face when there is no quantization"
    assert hf_model_dir is not None
    ## only load and call smooth quant routine once for all ranks
    hf_config = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_dir,
        device_map='auto',
        torch_dtype='auto' if not use_smooth_quant else torch.float16,
        trust_remote_code=True).half()

    os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
        "TOKENIZERS_PARALLELISM", "false")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir,
                                              trust_remote_code=True,
                                              use_fast=False,
                                              padding_side='left')
    dataset = load_calib_dataset(calib_dataset)

    system_prompt = "You are a useful assistant, please directly output the corresponding summary according to the article entered by the user."
    gen_config_path = os.path.join(hf_model_dir, 'generation_config.json')
    with open(gen_config_path, 'r') as f:
        gen_config = json.load(f)
    chat_format = getattr(gen_config, 'chat_format', 'chatml')
    act_range = capture_activation_range(hf_model, config.qwen_type, tokenizer,
                                         dataset, system_prompt, chat_format)
    qkv_para = {}
    # smoother for inputs of self_attn.o_proj and mlp.down_proj
    smoother = {}
    if use_smooth_quant:
        if config.qwen_type == 'qwen':
            smooth_qwen_model(hf_model, act_range, quant_config.smoothquant_val,
                              qkv_para, smoother)
        else:
            smooth_qwen2_model(hf_model, act_range,
                               quant_config.smoothquant_val, qkv_para, smoother)

    for rank in range(mapping.world_size):
        # To avoid changing the mapping arg in-place, also the given mapping from caller is rank agnostic, since quantize is called from only one rank
        config = copy.deepcopy(config)
        config.set_rank(rank)
        weights = load_weights_from_hf_model(hf_model,
                                             config=config,
                                             act_range=act_range,
                                             qkv_para=qkv_para,
                                             smoother=smoother)
        safetensors.torch.save_file(
            weights, os.path.join(output_dir, f'rank{rank}.safetensors'))
        del weights


def load_weights_from_hf_model(hf_model,
                               config: QWenConfig,
                               act_range: Optional[dict] = None,
                               qkv_para: Optional[dict] = None,
                               smoother: Optional[dict] = None):
    #TODO: simplify the parameters here

    assert hf_model is not None
    plugin_weight_only_quant_type = None  # the value does not matter when use_weight_only is False
    quant_algo = config.quantization.quant_algo
    if quant_algo == QuantAlgo.W8A16:
        plugin_weight_only_quant_type = torch.int8
    elif quant_algo == QuantAlgo.W4A16:
        plugin_weight_only_quant_type = torch.quint4x2
    else:
        plugin_weight_only_quant_type = None
    use_gemm_woq_plugin = (not config.disable_weight_only_quant_plugin)

    mapping = config.mapping
    moe_config = config.moe

    use_weight_only = quant_algo in [QuantAlgo.W8A16, QuantAlgo.W4A16]
    use_smooth_quant = config.quantization.use_plugin_sq
    per_channel = use_smooth_quant and 'PER_CHANNEL' in quant_algo
    per_token = use_smooth_quant and 'PER_TOKEN' in quant_algo
    int8_kv_cache = config.quantization.kv_cache_quant_algo == QuantAlgo.INT8
    qwen_type = config.qwen_type
    weights = convert_hf_qwen(
        hf_model,
        qwen_type,
        mapping,
        vocab_size=config.vocab_size,
        dtype=config.dtype,
        use_weight_only=use_weight_only,
        use_gemm_woq_plugin=use_gemm_woq_plugin,
        plugin_weight_only_quant_type=plugin_weight_only_quant_type,
        use_parallel_embedding=config.use_parallel_embedding,
        sharding_dim=config.embedding_sharding_dim,
        share_embedding_table=config.share_embedding_table,
        use_smooth_quant=use_smooth_quant,
        per_channel=per_channel,
        per_token=per_token,
        int8_kv_cache=int8_kv_cache,
        act_range=act_range,
        qkv_para=qkv_para,
        smoother=smoother,
        moe_config=moe_config)
    return weights


def load_weights_from_hf_gptq_model(hf_model, config: QWenConfig):
    logger.info("loading weights from groupwise GPTQ QWen safetensors...")
    weights = {}
    tik = time.time()

    qwen_type = config.qwen_type
    num_hidden_layers = config.num_hidden_layers
    mapping = config.mapping
    dtype = config.dtype

    model_params = {k: v for k, v in hf_model.state_dict().items()}
    torch.cuda.empty_cache()
    valid_types = ('qwen', 'qwen2')
    assert qwen_type in valid_types, f"Unsupported Qwen type: {qwen_type}, only {valid_types} are supported for GPTQ."
    layer_prefix = "transformer.h." if qwen_type == 'qwen' else "model.layers."
    key_list = get_qwen_key_list(qwen_type)

    def torch_split(v, dim):
        if v.shape[dim] % mapping.tp_size != 0:
            logger.error(
                "Current weight shape is invalid for mapping.tp_size=" +
                str(mapping.tp_size))
            assert False, "Invalid TP size"
        return v.split(v.shape[dim] // mapping.tp_size,
                       dim=dim)[mapping.tp_rank]

    def unpack_int32_into_int8(w_packed):
        # unpack inputs packed in int32/float32 into uint4 and store them in int8 format
        w_packed_int4x2 = w_packed.contiguous().view(torch.uint8)
        w_unpacked = torch.zeros(w_packed_int4x2.shape[0],
                                 w_packed_int4x2.shape[1] * 2,
                                 dtype=torch.int8)
        w_unpacked[:, ::2] = w_packed_int4x2 % 16
        w_unpacked[:, 1::2] = w_packed_int4x2 // 16
        return w_unpacked.contiguous()

    def process_and_assign_weight(v: List[torch.Tensor],
                                  tllm_prex: str,
                                  tp_dim: int = -1):
        if tp_dim == -1:
            qweight_int32, qzeros_int32, scales_fp16 = [
                item.cpu() for item in v
            ]
        else:
            qweight_int32, qzeros_int32, scales_fp16 = [
                torch_split(item, tp_dim).cpu() for item in v
            ]

        USE_UINT4_INPUT = 1  # Set to true if checkpoint store UINT4 weights
        USE_GPTQ_FOR_QWEN = 1  # GPTQ-for-QWEN added 1 to zeros

        qweight_unpacked_int8 = unpack_int32_into_int8(
            qweight_int32.T).T.contiguous() - 8
        qweight_interleaved = preprocessor(packer(qweight_unpacked_int8),
                                           torch.quint4x2,
                                           torch.float16).view(torch.float16)
        # zeros = zeros * scales
        qzeros_unpacked_int32 = unpack_int32_into_int8(qzeros_int32)
        if not USE_UINT4_INPUT:
            # Correcting UINT4 values back to INT4 order
            mask_negative = qzeros_unpacked_int32[qzeros_unpacked_int32 < 0]
            mask_positive = qzeros_unpacked_int32[qzeros_unpacked_int32 >= 0]
            qzeros_unpacked_int32 = qzeros_unpacked_int32 + 16 * mask_negative - 16 * mask_positive
        zeros_x_scales_fp16 = (-qzeros_unpacked_int32 + 8 * USE_UINT4_INPUT -
                               USE_GPTQ_FOR_QWEN) * scales_fp16
        zeros_x_scales_fp16 = zeros_x_scales_fp16.half()

        results = {
            f'{tllm_prex}.weight': qweight_interleaved,
            f'{tllm_prex}.weights_scaling_factor': scales_fp16,
            f'{tllm_prex}.zero': zeros_x_scales_fp16,
        }
        return results

    packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4
    preprocessor = torch.ops.trtllm.preprocess_weights_for_mixed_gemm
    torch_dtype = str_dtype_to_torch(dtype)

    # Load weights from GPTQ checkpoint into TRT-LLM module
    # 1. vocab_embedding
    v = model_params[key_list[7] + '.weight']
    if mapping.is_first_pp_rank():
        weights['transformer.vocab_embedding.weight'] = v.to(torch_dtype)

    # 2. ln_f
    v = model_params[key_list[8] + '.weight']
    if mapping.is_last_pp_rank():
        weights['transformer.ln_f.weight'] = v.to(torch_dtype)

    # 3. lm_head
    v = model_params['lm_head.weight']
    if mapping.is_last_pp_rank():
        weights['lm_head.weight'] = torch_split(v, 0).to(torch_dtype)

    # 4. Weights inside each layer
    layers_per_pipeline_stage = num_hidden_layers // mapping.pp_size
    layers_range = list(
        range(mapping.pp_rank * layers_per_pipeline_stage,
              (mapping.pp_rank + 1) * layers_per_pipeline_stage, 1))
    suffixs = [".qweight", ".qzeros", ".scales"]

    for l in tqdm(layers_range, desc="loading weight in each layer..."):
        layer_idx = l - mapping.pp_rank * layers_per_pipeline_stage
        prefix = layer_prefix + str(layer_idx) + "."
        tllm_prex = f'transformer.layers.{l-layers_range[0]}'
        # 4.1 attention.qkv
        qkv_weight_list = []
        if qwen_type == 'qwen':
            for suf in suffixs:
                qkv_part = model_params[prefix + key_list[0] + suf]
                q_emb = qkv_part.shape[1] // 3
                model_emb = qkv_part.shape[0]
                qkv_part = qkv_part.reshape(model_emb, 3, q_emb)
                qkv_part = torch_split(qkv_part, 2)
                qkv_part = qkv_part.reshape(model_emb,
                                            3 * (q_emb // mapping.tp_size))
                qkv_weight_list.append(qkv_part)
        else:
            for suf in suffixs:
                qkv_list = []
                for comp in ["q_proj", "k_proj", "v_proj"]:
                    comp_part = model_params[prefix + key_list[0] + comp + suf]
                    comp_part = torch_split(comp_part, 1)
                    qkv_list.append(comp_part)
                qkv_weight_list.append(torch.cat(qkv_list, dim=1))
        weights.update(
            process_and_assign_weight(qkv_weight_list,
                                      f'{tllm_prex}.attention.qkv'))
        # 4.2 attention.bias
        suf = ".bias"
        if qwen_type == 'qwen':
            qkv_bias = model_params[prefix + key_list[0] +
                                    suf].to(torch_dtype).cpu().contiguous()
            q_emb = qkv_bias.shape[0] // 3
            qkv_bias = qkv_bias.reshape(3, q_emb)
            split_v = split(qkv_bias, mapping.tp_size, mapping.rank, dim=1)
            qkv_bias = split_v.reshape(3 * (q_emb // mapping.tp_size))
        else:
            qkv_bias_list = []
            for comp in ["q_proj", "k_proj", "v_proj"]:
                comp_part = model_params[prefix + key_list[0] + comp + suf].to(
                    torch_dtype).cpu().contiguous()
                comp_part = torch_split(comp_part, dim=0)
                qkv_bias_list.append(comp_part)
            qkv_bias = torch.cat(qkv_bias_list, dim=0)
        weights[tllm_prex + ".attention.qkv.bias"] = qkv_bias
        # 4.3 attention.dense
        qkv_dense_list = []
        for suf in suffixs:
            qkv_dense_part = model_params[prefix + key_list[1] + suf]
            qkv_dense_list.append(qkv_dense_part)
        weights.update(
            process_and_assign_weight(qkv_dense_list,
                                      f'{tllm_prex}.attention.dense',
                                      tp_dim=0))
        # 4.4 mlp.gate
        mlp_gate_list = []
        for suf in suffixs:
            mlp_gate_part = model_params[prefix + key_list[2] + suf]
            mlp_gate_list.append(mlp_gate_part)
        weights.update(
            process_and_assign_weight(mlp_gate_list,
                                      f'{tllm_prex}.mlp.gate',
                                      tp_dim=1))
        # 4.5 mlp.fc
        mlp_fc_list = []
        for suf in suffixs:
            mlp_fc_part = model_params[prefix + key_list[3] + suf]
            mlp_fc_list.append(mlp_fc_part)
        weights.update(
            process_and_assign_weight(mlp_fc_list,
                                      f'{tllm_prex}.mlp.fc',
                                      tp_dim=1))
        # 4.6 mlp.proj
        mlp_proj_list = []
        for suf in suffixs:
            mlp_proj_part = model_params[prefix + key_list[4] + suf]
            mlp_proj_list.append(mlp_proj_part)
        weights.update(
            process_and_assign_weight(mlp_proj_list,
                                      f'{tllm_prex}.mlp.proj',
                                      tp_dim=0))
        # 4.7 input_layernorm
        v = model_params[prefix + key_list[5] + '.weight']
        weights[f'{tllm_prex}.input_layernorm.weight'] = v.to(torch_dtype)
        # 4.8 post_layernorm
        v = model_params[prefix + key_list[6] + '.weight']
        weights[f'{tllm_prex}.post_layernorm.weight'] = v.to(torch_dtype)

    tok = time.time()
    t = time.strftime("%H:%M:%S", time.gmtime(tok - tik))
    logger.info(f"weights loaded. total time: {t}")

    return weights

def load_torch_meta_ckpt(meta_ckpt_path: Path):
    '''
        meta_ckpt_path: The format of meta_ckpt_path is like <xxx>/consolidated.xx There are two possible cases:
            1. A file like <xxx>/consolidated.xx.pth, loading it by torch.load directly
            2. A folder like <xxx>/consolidated.xx/, need to load all weights in the folder.
    '''
    file_path = meta_ckpt_path.parent / (meta_ckpt_path.name + ".pth")
    if file_path.exists() and file_path.is_file():
        return torch.load(file_path, map_location="cpu")
    else:
        folder_path = meta_ckpt_path
        assert folder_path.exists() and folder_path.is_dir()

        ckpts = list(Path(folder_path).glob("consolidated-*.pth"))

        all_weights = {}
        for ckpt in ckpts:
            _weight = torch.load(ckpt, map_location="cpu")
            all_weights = all_weights | _weight
            del _weight
        return all_weights

def fp8_per_channel_quant_weight_gpu(weight, clamp_val, rank=0):
    weight = weight.to("cuda:" + str(rank))
    # activation range bound.
    x = weight.to(torch.float32).clamp(clamp_val[0], clamp_val[1])
    xmax = x.abs().max(-1, keepdim=True).values
    # minimum scaling factor.
    torch_weight_scales = (xmax / 448.0).clamp(min=1.0 / (448.0 * 512.0))
    out = x / torch_weight_scales
    torch_weight_scales = torch_weight_scales.reshape(-1)
    out = torch.clamp(out, -448, 448)
    processed_torch_weights = out.to(torch.float8_e4m3fn)

    processed_torch_weights = processed_torch_weights.to(
        torch.float8_e4m3fn).cpu()
    torch_weight_scales = torch_weight_scales.cpu()

    return processed_torch_weights, torch_weight_scales

def load_weights_from_meta_ckpt(meta_ckpt_dir: str, config: QWenConfig):
    torch_dtype = str_dtype_to_torch(config.dtype)
    mapping = config.mapping
    use_fp8_rowwise = config.quant_mode.has_fp8_rowwise()
    if config.quant_mode.has_any_quant() and not use_fp8_rowwise:
        logger.error(
            "Meta ckpts only support fp8_rowwise quantization currently.")
    weights = {}
    # Meta's recipe of not using fp8 rowwise for the first and last layer.
    exclude_layers_id = [0, config.num_hidden_layers - 1]

    def gather_ckpts(ckpts):
        gathered = {}
        for k in ckpts[0]:
            d = 0
            # TODO(bhsueh) not sure should we consider tok here.
            if any([n in k for n in ["wo", "w2"]]):
                d = 1
            if "norm" in k or "rope" in k:  # no TP
                gathered[k] = ckpts[0][k].clone()
            else:
                gathered[k] = torch.cat([pt[k] for pt in ckpts], dim=d).clone()
        return gathered

    def split_ckpt(ckpt, ranks_per_ckpt, ckpt_rank):
        split_ckpt = {}
        for k, v in ckpt.items():
            d = 0
            if any(n in k for n in
                   ["wo", "feed_forward.w2", "tok", "feed_forward.gate"]):
                d = 1
            if "norm" in k or "rope" in k:  # no TP
                split_ckpt[k] = v.clone()
            elif config.num_key_value_heads < mapping.tp_size and any(
                    n in k for n in ["wk", "wv"]):
                assert mapping.tp_size % config.num_key_value_heads == 0
                # special case: we need to duplicate KV head
                tmp = dup_kv_weight(v, config.num_key_value_heads,
                                    mapping.tp_size)
                split_ckpt[k] = torch.split(tmp,
                                            tmp.shape[d] // ranks_per_ckpt,
                                            dim=d)[ckpt_rank].clone()
            else:
                split_ckpt[k] = torch.split(v,
                                            v.shape[d] // ranks_per_ckpt,
                                            dim=d)[ckpt_rank].clone()
        return split_ckpt

    def get_current_weights(num_ckpts):
        if num_ckpts > mapping.tp_size:
            # combine ckpts
            assert (num_ckpts % mapping.tp_size) == 0
            nf = num_ckpts // mapping.tp_size
            fs = nf * mapping.tp_rank
            file_ids = list(range(fs, fs + nf))
            ckpts = []
            for f in file_ids:
                ckpt = load_torch_meta_ckpt(
                    Path(meta_ckpt_dir, f"consolidated.{f:02d}"))
                ckpts.append(ckpt)
            return gather_ckpts(ckpts)
        elif num_ckpts < mapping.tp_size:
            # split ckpt
            assert (mapping.tp_size % num_ckpts) == 0
            ranks_per_ckpt = mapping.tp_size // num_ckpts
            ckpt_fid = mapping.tp_rank // ranks_per_ckpt
            ckpt_rank = mapping.tp_rank % ranks_per_ckpt
            nH_per_ckpt = config.num_attention_heads // num_ckpts
            assert (nH_per_ckpt % ranks_per_ckpt) == 0
            ckpt = load_torch_meta_ckpt(
                Path(meta_ckpt_dir, f"consolidated.{ckpt_fid:02d}"))
            return split_ckpt(ckpt, ranks_per_ckpt, ckpt_rank)

        # num_ckpts == tensor_parallel, 1:1 mapping from files to TP
        return load_torch_meta_ckpt(
            Path(meta_ckpt_dir, f"consolidated.{mapping.tp_rank:02d}"))

    def permute(w, nH, d, dH):
        # due to MQA's wk, nH*dH != d could be true
        return w.view(nH, dH // 2, 2, d).transpose(1, 2).reshape(nH * dH, d)

    def extract_layer_idx(name):
        ss = name.split('.')
        for s in ss:
            if s.isdigit():
                return s
        return None

    if not hasattr(load_weights_from_meta_ckpt, "saved_embed"):
        load_weights_from_meta_ckpt.saved_embed = None

    def combine_embeddings(embeds, num_ckpts):
        if len(embeds) == 1:
            return embeds[0]
        assert [
            embeds[i].shape == embeds[i + 1].shape
            for i in range(len(embeds) - 1)
        ]
        if embeds[0].shape[0] == config.vocab_size // num_ckpts:
            merge_dim = 0
        elif embeds[0].shape[1] == config.hidden_size // num_ckpts:
            merge_dim = 1
        else:
            logger.error("Unable to infer embedding split dimension")
            assert False, "Unable to infer embedding split dimension"
        return torch.cat(embeds, dim=merge_dim)

    def gather_embedding(cur_embed, name: str, num_ckpts):
        if mapping.tp_size == 1:
            # even if num_ckpts > 1, get_current_weights will already have it gathered
            return cur_embed
        if load_weights_from_meta_ckpt.saved_embed is None:
            embeds = [None] * num_ckpts
            for i in range(num_ckpts):
                ckpt = load_torch_meta_ckpt(
                    Path(meta_ckpt_dir, f"consolidated.{i:02d}"))
                embeds[i] = ckpt[name]
            embed = combine_embeddings(embeds, num_ckpts).to(torch_dtype)
            load_weights_from_meta_ckpt.saved_embed = embed

        return load_weights_from_meta_ckpt.saved_embed

    logger.info('Loading weights from Meta LLaMA checkpoints ...')
    tik = time.time()

    num_kv_heads = config.num_key_value_heads
    mha_mode = (num_kv_heads == config.num_attention_heads)

    ckpts = list(Path(meta_ckpt_dir).glob("consolidated.*"))
    num_ckpts = len(ckpts)
    # llama/llama2 doesn't have MQA. So, simplifying loader logic by not worrying about it.
    assert num_kv_heads > 1 or num_kv_heads >= num_ckpts, \
        f"We don't know how the {num_kv_heads} KV heads are distributed among {num_ckpts} checkpoints."

    tik = time.time()
    ckpt = get_current_weights(num_ckpts)
    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'[{mapping.rank}] get_current_weights. Total time: {t}')

    head_size = config.hidden_size // config.num_attention_heads
    layers_range = mapping.pp_layers(config.num_hidden_layers)

    for l in layers_range:
        prefix = f'layers.{l}.attention.'
        q_weight = permute(ckpt[prefix + 'wq.weight'].clone(),
                           nH=(config.num_attention_heads // mapping.tp_size),
                           d=config.hidden_size,
                           dH=head_size)
        if num_kv_heads < mapping.tp_size and num_ckpts >= mapping.tp_size:
            assert mapping.tp_size % num_kv_heads == 0
            assert False, "Not supported yet"
        k_weight = permute(ckpt[prefix + 'wk.weight'].clone(),
                           nH=((num_kv_heads + mapping.tp_size - 1) //
                               mapping.tp_size),
                           d=config.hidden_size,
                           dH=head_size)
        v_weight = ckpt[prefix + 'wv.weight'].clone()

        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
        ckpt[prefix + 'qkv.weight'] = qkv_weight

    for k, v in tqdm(ckpt.items()):
        dtype = torch_dtype if 'feed_forward.gate' not in k else torch.float32

        v = v.to(dtype)
        if "tok_embeddings" in k:
            if not config.use_parallel_embedding:
                v = gather_embedding(v, k, num_ckpts)
            elif config.embedding_sharding_dim == 0:
                # this needs a gather and then resplit along different dims
                v = gather_embedding(v, k, num_ckpts)
                v = split(v, mapping.tp_size, mapping.tp_rank, 0)
            if mapping.is_first_pp_rank():
                weights['transformer.vocab_embedding.weight'] = v
        elif "output" in k:
            if mapping.is_last_pp_rank():
                if config.vocab_size % mapping.tp_size != 0:
                    # padding
                    vocab_size_padded = pad_vocab_size(config.vocab_size,
                                                       mapping.tp_size)
                    pad_width = vocab_size_padded - config.vocab_size
                    v = torch.from_numpy(
                        np.pad(v.detach().cpu().numpy(),
                               ((0, pad_width), (0, 0)),
                               'constant',
                               constant_values=0))
                weights['lm_head.weight'] = v.detach().clone()
        elif k == "norm.weight":
            if mapping.is_last_pp_rank():
                weights['transformer.ln_f.weight'] = v
        else:
            # layer specific weights
            layer_idx = extract_layer_idx(k)
            if layer_idx is None or int(layer_idx) not in layers_range:
                continue

            # Meta's recipe of not using fp8 rowwise for the first and last layer.
            use_fp8_rowwise_in_layer = use_fp8_rowwise and (
                int(layer_idx) not in exclude_layers_id)
            idx = int(layer_idx) - layers_range[0]
            tllm_prex = f'transformer.layers.{idx}.'

            if 'attention_norm.weight' in k:
                weights[tllm_prex + 'input_layernorm.weight'] = v
            elif 'ffn_norm.weight' in k:
                weights[tllm_prex + 'post_layernorm.weight'] = v
            elif 'feed_forward.w3.weight' in k:
                if use_fp8_rowwise_in_layer:
                    processed_torch_weights, torch_weight_scales = fp8_per_channel_quant_weight_gpu(
                        v, config.quantization.clamp_val)
                    weights[tllm_prex +
                            'mlp.gate.weight'] = processed_torch_weights
                    weights[tllm_prex +
                            'mlp.gate.per_channel_scale'] = torch_weight_scales
                else:
                    weights[tllm_prex + 'mlp.gate.weight'] = v
            elif 'feed_forward.w2.weight' in k:
                if use_fp8_rowwise_in_layer:
                    processed_torch_weights, torch_weight_scales = fp8_per_channel_quant_weight_gpu(
                        v, config.quantization.clamp_val)
                    weights[tllm_prex +
                            'mlp.proj.weight'] = processed_torch_weights
                    weights[tllm_prex +
                            'mlp.proj.per_channel_scale'] = torch_weight_scales
                else:
                    weights[tllm_prex + 'mlp.proj.weight'] = v
            elif 'feed_forward.w1.weight' in k:
                if use_fp8_rowwise_in_layer:
                    processed_torch_weights, torch_weight_scales = fp8_per_channel_quant_weight_gpu(
                        v, config.quantization.clamp_val)
                    weights[tllm_prex +
                            'mlp.fc.weight'] = processed_torch_weights
                    weights[tllm_prex +
                            'mlp.fc.per_channel_scale'] = torch_weight_scales
                else:
                    weights[tllm_prex + 'mlp.fc.weight'] = v
            elif 'attention.wo.weight' in k:
                weights[tllm_prex + 'attention.dense.weight'] = v
            elif 'attention.qkv.weight' in k:
                weights[tllm_prex + 'attention.qkv.weight'] = v
                print('Attention weight & dtype', v.shape, v.dtype)
            elif 'feed_forward.gate' in k:
                weights[tllm_prex + 'mlp.router.weight'] = v

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Weights loaded. Total time: {t}')
    return weights

def load_weights_from_lmquant(lmquant_ckpt_path: str, config: QWenConfig):
    logger.info(
        'Loading weights from lmquant torch checkpoint for QServe W4A8 inference...'
    )
    weights = {}
    tik = time.time()

    per_group = config.quant_mode.has_per_group_scaling()
    group_size = 128 if per_group else -1

    num_hidden_layers = config.num_hidden_layers
    vocab_size = config.vocab_size
    dtype = config.dtype
    mapping = config.mapping
    torch_dtype = str_dtype_to_torch(dtype)
    assert torch_dtype == torch.float16, "Currently QServe only supports float16"

    # weight
    fake_quant_weights = torch.load(lmquant_ckpt_path + '/model.pt',
                                    map_location='cpu')
    # scale.0, scale.1, zero
    quant_params = torch.load(lmquant_ckpt_path + '/scale.pt',
                              map_location='cpu')
    print(50 * '--')
    print('\n'.join(list(fake_quant_weights.keys())[:40])  )
    print(50 * '--')
    print('\n'.join(list(quant_params.keys())[:40])  )
    print(50 * '--')
    def load(key):
        if 'zero' in key:
            v = quant_params[key]
            # https://github.com/mit-han-lab/qserve/blob/64ee627dfd747510809998d3592439f05a71ba31/scripts/ckpt_converter/checkpoint_converter.py#L99
            # print(key, v.min(), v.max())
            if v.min() < 0:
                v = v + 8
            return v
        if 'scale' in key:
            return quant_params[key]
        return fake_quant_weights[key]

    if per_group:
        lmquant_suffix = [
            'weight', 'weight.scale.0', 'weight.scale.1', 'weight.scaled_zero' # 'weight.zero'
        ]
        qserve_suffix = ['weight', 's1_scales', 's2_scales', 's2_zeros']
    else:
        lmquant_suffix = ['weight', 'weight.scale.0',  'weight.scaled_zero']# 'weight.zero']
        qserve_suffix = ['weight', 's1_scales', 's1_szeros']

    def tp_split_tensor(v: torch.Tensor, dim):
        if v.shape[dim] % mapping.tp_size != 0:
            logger.error(
                f"Current weight shape is invalid for mapping.tp_size={mapping.tp_size}"
            )
            assert False, "Invalid TP size"
        return v.split(v.shape[dim] // mapping.tp_size,
                       dim=dim)[mapping.tp_rank].contiguous()

    def tp_split_weight_and_params(v: List[torch.Tensor], column_linear: bool):
        if per_group:
            weight, s1_scales, s2_scales, s2_zeros = v
            # weight (out_features, in_features)
            # weight.scale.0 (out_features, 1, 1, 1)
            # weight.scale.1 (out_features, 1, in_features/group_size, 1)
            # weight.zero (out_features, 1, in_features/group_size, 1)
            if column_linear:
                weight = tp_split_tensor(weight, 0)
                s1_scales = tp_split_tensor(s1_scales, 0)
                s2_scales = tp_split_tensor(s2_scales, 0)
                s2_zeros = tp_split_tensor(s2_zeros, 0)
            else:
                weight = tp_split_tensor(weight, 1)
                s1_scales = s1_scales
                s2_scales = tp_split_tensor(s2_scales, 2)
                s2_zeros = tp_split_tensor(s2_zeros, 2)
            return [weight, s1_scales, s2_scales, s2_zeros]
        else:
            weight, s1_scales, s1_zeros = v
            # weight (out_features, in_features)
            # weight.scale.0 (out_features, 1, 1, 1)
            # weight.zero (out_features, 1, 1, 1)
            if column_linear:
                weight = tp_split_tensor(weight, 0)
                s1_scales = tp_split_tensor(s1_scales, 0)
                s1_zeros = tp_split_tensor(s1_zeros, 0)
            else:
                weight = tp_split_tensor(weight, 1)
                s1_scales = s1_scales
                s1_zeros = s1_zeros
            return [weight, s1_scales, s1_zeros]

    def process_weight_and_params(v: List[torch.Tensor], tllm_prex: str):
        if per_group:
            weight, s1_scales, s2_scales, s2_zeros = v
            qweight = qserve_quantize_weight_per_group(weight, s1_scales,
                                                       s2_scales, s2_zeros,
                                                       group_size)
            qweight, s1_scales, s2_scales, s2_zeros = qserve_pack_reorder_per_group(
                qweight, s1_scales, s2_scales, s2_zeros, group_size)

            return {
                # Note: Linear modules in TRTLLM do not use the name 'qweight'
                f'{tllm_prex}.{qserve_suffix[0]}': qweight,
                f'{tllm_prex}.{qserve_suffix[1]}': s1_scales,
                f'{tllm_prex}.{qserve_suffix[2]}': s2_scales,
                f'{tllm_prex}.{qserve_suffix[3]}': s2_zeros,
            }
        else:
            weight, s1_scales, s1_zeros = v
            qweight = qserve_quantize_weight_per_channel(
                weight, s1_scales, s1_zeros)
            qweight, s1_scales, s1_szeros = qserve_pack_reorder_per_channel(
                qweight, s1_scales, s1_zeros)
            return {
                # Note: Linear modules in TRTLLM use the name 'weight' instead of 'qweight'
                f'{tllm_prex}.{qserve_suffix[0]}': qweight,
                f'{tllm_prex}.{qserve_suffix[1]}': s1_scales,
                f'{tllm_prex}.{qserve_suffix[2]}': s1_szeros,
            }

    # Load weights
    # 1. vocab_embedding
    v = load('model.embed_tokens.weight')
    if mapping.is_first_pp_rank():
        weights['transformer.vocab_embedding.weight'] = v.to(torch_dtype)

    # 2. lm_head
    v = load('lm_head.weight')
    if mapping.is_last_pp_rank():
        if vocab_size % mapping.tp_size != 0:
            # padding
            vocab_size_padded = pad_vocab_size(vocab_size, mapping.tp_size)
            pad_width = vocab_size_padded - vocab_size
            v = torch.nn.functional.pad(v, (0, 0, 0, pad_width))
        weights['lm_head.weight'] = tp_split_tensor(v, 0).to(torch_dtype)

    # 3. ln_f
    v = load('model.norm.weight')
    if mapping.is_last_pp_rank():
        weights['transformer.ln_f.weight'] = v.to(torch_dtype)

    # 4. Weights inside each layer
    layers_range = mapping.pp_layers(num_hidden_layers)
    for layer_idx in layers_range:
        prefix = f'model.layers.{layer_idx}'
        logger.info(f'Processing weights in layer: {layer_idx}')
        tllm_prex = f'transformer.layers.{layer_idx - layers_range[0]}'

        # 4.1 attention.qkv
        qkv_list = []
        for comp in ["q", "k", "v"]:
            #for suffix in lmquant_suffix:
            #    print(f'Load -- {prefix}.self_attn.{comp}_proj.{suffix}')
            v = [
                load(f'{prefix}.self_attn.{comp}_proj.{suffix}')
                for suffix in lmquant_suffix
            ]
            v = tp_split_weight_and_params(v, column_linear=True)
            qkv_list.append(v)
        # Concat qkv
        q, k, v = qkv_list
        qkv = [
            torch.concat((q[i], k[i], v[i]), dim=0)
            for i in range(len(lmquant_suffix))
        ]
        weights.update(
            process_weight_and_params(qkv, f'{tllm_prex}.attention.qkv'))
        # tmp_key = f'{tllm_prex}.attention.qkv'

        # print(tmp_key, weights[tmp_key].shape, weights[tmp_key].dtype)
        # 4.2 attention.dense
        v = [
            load(f'{prefix}.self_attn.o_proj.{suffix}')
            for suffix in lmquant_suffix
        ]
        v = tp_split_weight_and_params(v, column_linear=False)
        weights.update(
            process_weight_and_params(v, f'{tllm_prex}.attention.dense'))

        # TODO: The naming here is tricky.
        # The implementation of GatedMLP is act(fc(x)) * gate(x).
        # However, the common convention is act(gate_proj(x)) * up_proj(x).

        # 4.3 mlp.gate
        v = [
            load(f'{prefix}.mlp.up_proj.{suffix}') for suffix in lmquant_suffix
        ]
        v = tp_split_weight_and_params(v, column_linear=True)
        weights.update(process_weight_and_params(v, f'{tllm_prex}.mlp.gate'))

        # 4.4 mlp.fc
        v = [
            load(f'{prefix}.mlp.gate_proj.{suffix}')
            for suffix in lmquant_suffix
        ]
        v = tp_split_weight_and_params(v, column_linear=True)
        weights.update(process_weight_and_params(v, f'{tllm_prex}.mlp.fc'))

        # 4.5 mlp.proj
        v = [
            load(f'{prefix}.mlp.down_proj.{suffix}')
            for suffix in lmquant_suffix
        ]
        v = tp_split_weight_and_params(v, column_linear=False)
        weights.update(process_weight_and_params(v, f'{tllm_prex}.mlp.proj'))

        # 4.6 input_layernorm
        v = load(f'{prefix}.input_layernorm.weight')
        weights[f'{tllm_prex}.input_layernorm.weight'] = v.to(torch_dtype)

        # 4.7 post_layernorm
        v = load(f'{prefix}.post_attention_layernorm.weight')
        weights[f'{tllm_prex}.post_layernorm.weight'] = v.to(torch_dtype)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    logger.info(f'Weights loaded. Total time: {t}')

    # TODO: All the RMSNorm weight, including ln_f, input_layernorm, post_layernorm, are actually all 1s
    # Could implement a simplified module without weight
    return weights
