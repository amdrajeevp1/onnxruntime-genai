# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import json
import os

import numpy as np

from .mistral import MistralModel


class GemmaModel(MistralModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.embed_attrs["scale"] = np.round(np.sqrt(self.hidden_size), decimals=2)
        self.layernorm_attrs["add_offset"] = 1


class Gemma2Model(GemmaModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.layernorm_attrs["cast"]["use_fp32"] = True
        self.layernorm_attrs["cast"]["root_input"] = True
        self.layernorm_attrs["cast"]["skip_input"] = False
        self.layernorm_attrs["cast"]["output_0"] = True
        self.layernorm_attrs["cast"]["output_3"] = False
        self.attention_attrs["scale"] = config.query_pre_attn_scalar**-0.5

    def is_local(self, layer_id):
        return layer_id % 2 == 1

    def make_layernorm(self, layer_id, layernorm, skip, simple, location):
        if "final_norm" in location:
            # Set cast for final LayerNorm since it is a special case and not covered in `make_layer`
            self.layernorm_attrs["cast"]["root_input"] = False
        super().make_layernorm(layer_id, layernorm, skip, simple, location)

    def make_layer(self, layer_id, layer):
        # Gemma-2 decoder layer is typically defined as:
        # input_layernorm --> attention --> post_attention_layernorm --> pre_ffn_layernorm --> MLP --> post_ffn_layernorm

        # Adjust LayerNorm attributes because of extra LayerNorms inserted
        # 1. Only cast root_input if the first layer of LayerNorms are being created
        original_cast_root_input = self.layernorm_attrs["cast"]["root_input"]
        self.layernorm_attrs["cast"]["root_input"] = self.layernorm_attrs["first_layernorm"]
        self.make_layernorm(
            layer_id,
            layer.input_layernorm,
            skip=not self.layernorm_attrs["first_layernorm"],
            simple=self.layernorm_attrs["simple"],
            location="input",
        )
        self.layernorm_attrs["cast"]["root_input"] = original_cast_root_input

        self.make_attention(layer_id, layer.self_attn, root_input=self.layernorm_attrs["output_0"])

        # Adjust LayerNorm attributes for extra LayerNorm to insert
        # 1. Temporarily set root_input for LayerNorm to skip_input for post_attention_layernorm
        # 2. Set skip_input to output of post_attention_layernorm
        # 3. Do not cast outputs from post_attention_layernorm
        original_root_input = self.layernorm_attrs["root_input"]
        original_cast_output_0 = self.layernorm_attrs["cast"]["output_0"]
        self.layernorm_attrs["root_input"] = self.layernorm_attrs["skip_input"]
        self.layernorm_attrs["cast"]["output_0"] = False
        self.make_layernorm(
            layer_id,
            layer.post_attention_layernorm,
            skip=False,
            simple=self.layernorm_attrs["simple"],
            location="post_attention",
        )
        self.layernorm_attrs["root_input"] = original_root_input
        self.layernorm_attrs["skip_input"] = self.layernorm_attrs["output_0"]
        self.layernorm_attrs["cast"]["output_0"] = original_cast_output_0

        # Adjust LayerNorm attributes because of extra LayerNorms inserted
        # 1. Only cast root_input if the first layer of LayerNorms are being created
        original_cast_root_input = self.layernorm_attrs["cast"]["root_input"]
        self.layernorm_attrs["cast"]["root_input"] = self.layernorm_attrs["first_layernorm"]
        self.make_layernorm(
            layer_id,
            layer.pre_feedforward_layernorm,
            skip=True,
            simple=self.layernorm_attrs["simple"],
            location="pre_feedforward",
        )
        self.layernorm_attrs["cast"]["root_input"] = original_cast_root_input

        self.make_mlp(layer_id, layer.mlp, root_input=self.layernorm_attrs["output_0"])

        # Adjust LayerNorm attributes for extra LayerNorm to insert
        # 1. Temporarily set root_input for LayerNorm to skip_input for post_feedforward_layernorm
        # 2. Set skip_input to output of post_feedforward_layernorm
        # 3. Do not cast outputs from post_feedforward_layernorm
        original_root_input = self.layernorm_attrs["root_input"]
        original_cast_output_0 = self.layernorm_attrs["cast"]["output_0"]
        self.layernorm_attrs["root_input"] = self.layernorm_attrs["skip_input"]
        self.layernorm_attrs["cast"]["output_0"] = False
        self.make_layernorm(
            layer_id,
            layer.post_feedforward_layernorm,
            skip=False,
            simple=self.layernorm_attrs["simple"],
            location="post_feedforward",
        )
        self.layernorm_attrs["root_input"] = original_root_input
        self.layernorm_attrs["skip_input"] = self.layernorm_attrs["output_0"]
        self.layernorm_attrs["cast"]["output_0"] = original_cast_output_0

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            # Norm after last decoder layer of model (last layer --> norm)
            self.layernorm_attrs["last_layernorm"] = True

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        original_window_size = self.window_size
        self.window_size = (
            original_window_size if self.is_local(layer_id) else -1
        )  # default is -1 in GroupQueryAttention kernel
        super().make_attention(layer_id, attention, root_input, **kwargs)
        self.window_size = original_window_size


class Gemma3Model(Gemma2Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        self.rope_local_theta = config.rope_local_base_freq
        self.make_rotary_embedding_multi_cache()

    def is_local(self, layer_id):
        return bool((layer_id + 1) % 6)

    def make_attention_init(self):
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True
        super().make_attention_init()

    def make_rotary_embedding_multi_cache(self):
        self.cos_cache_global_name, self.sin_cache_global_name = "cos_cache_global", "sin_cache_global"
        super().make_rotary_embedding_caches(
            cos_cache_name=self.cos_cache_global_name, sin_cache_name=self.sin_cache_global_name
        )

        # Create the new cos/sin caches for local attention layers with its own theta value
        self.rope_attrs["create_caches"] = True
        self.rope_attrs["theta"] = self.rope_local_theta

        self.cos_cache_local_name, self.sin_cache_local_name = "cos_cache_local", "sin_cache_local"
        super().make_rotary_embedding_caches(
            cos_cache_name=self.cos_cache_local_name, sin_cache_name=self.sin_cache_local_name
        )

    def make_rotary_embedding_caches(self, **kwargs):
        cos_cache_name = kwargs.get(
            "cos_cache_name", self.cos_cache_global_name if self.window_size == -1 else self.cos_cache_local_name
        )
        sin_cache_name = kwargs.get(
            "sin_cache_name", self.sin_cache_global_name if self.window_size == -1 else self.sin_cache_local_name
        )
        return super().make_rotary_embedding_caches(cos_cache_name=cos_cache_name, sin_cache_name=sin_cache_name)


class Gemma4Model(Gemma2Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        if extra_options.get("exclude_embeds", False):
            # Gemma-4 conditional models can nest text embeddings differently, so
            # we pre-wire the decoder input when embeds are excluded.
            self.layernorm_attrs["root_input"] = "inputs_embeds"
            self.layernorm_attrs["skip_input"] = "inputs_embeds"
        self.layer_types = config.layer_types
        self.rope_params_by_type = config.rope_parameters
        self.local_head_dim = config.head_dim if hasattr(config, "head_dim") else self.head_size
        self.global_head_dim = (
            config.global_head_dim if hasattr(config, "global_head_dim") and config.global_head_dim else self.local_head_dim
        )
        self.head_size_by_layer = [
            self.global_head_dim if layer_type == "full_attention" else self.local_head_dim
            for layer_type in self.layer_types
        ]
        self.num_kv_shared_layers = config.num_kv_shared_layers if hasattr(config, "num_kv_shared_layers") else 0
        self.hidden_size_per_layer_input = (
            config.hidden_size_per_layer_input if hasattr(config, "hidden_size_per_layer_input") else None
        )
        self.use_double_wide_mlp = config.use_double_wide_mlp if hasattr(config, "use_double_wide_mlp") else False
        # Gemma-4 attention modules expose their own runtime scaling (typically 1.0).
        # Do not force Gemma2/3-style query_pre_attn_scalar scaling globally.
        self.attention_attrs["scale"] = 1.0
        self.make_rotary_embedding_multi_cache()

    def is_local(self, layer_id):
        return self.layer_types[layer_id] == "sliding_attention"

    def make_attention_init(self):
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True
        super().make_attention_init()

    def _set_rope_profile(self, layer_type):
        rope_params = self.rope_params_by_type.get(layer_type, {})
        partial_rotary_factor = rope_params.get("partial_rotary_factor", 1.0)
        self.rope_attrs["theta"] = rope_params.get("rope_theta", self.rope_attrs["theta"])
        self.rope_attrs["partial_rotary_factor"] = partial_rotary_factor
        self.rope_attrs["rotary_embedding_dim"] = (
            int(self.head_size * partial_rotary_factor) if partial_rotary_factor != 1.0 else 0
        )

    def make_rotary_embedding_multi_cache(self):
        original_head_size = self.head_size
        original_theta = self.rope_attrs["theta"]
        original_partial_rotary_factor = self.rope_attrs["partial_rotary_factor"]
        original_rotary_embedding_dim = self.rope_attrs["rotary_embedding_dim"]
        original_create_caches = self.rope_attrs["create_caches"]

        self.cos_cache_local_name, self.sin_cache_local_name = "cos_cache_local", "sin_cache_local"
        self.head_size = self.local_head_dim
        self._set_rope_profile("sliding_attention")
        self.rope_attrs["create_caches"] = True
        super().make_rotary_embedding_caches(
            cos_cache_name=self.cos_cache_local_name, sin_cache_name=self.sin_cache_local_name
        )

        self.cos_cache_global_name, self.sin_cache_global_name = "cos_cache_global", "sin_cache_global"
        self.head_size = self.global_head_dim
        self._set_rope_profile("full_attention")
        self.rope_attrs["create_caches"] = True
        super().make_rotary_embedding_caches(
            cos_cache_name=self.cos_cache_global_name, sin_cache_name=self.sin_cache_global_name
        )

        self.head_size = original_head_size
        self.rope_attrs["theta"] = original_theta
        self.rope_attrs["partial_rotary_factor"] = original_partial_rotary_factor
        self.rope_attrs["rotary_embedding_dim"] = original_rotary_embedding_dim
        self.rope_attrs["create_caches"] = original_create_caches

    def make_rotary_embedding_caches(self, **kwargs):
        cos_cache_name = kwargs.get(
            "cos_cache_name", self.cos_cache_global_name if self.window_size == -1 else self.cos_cache_local_name
        )
        sin_cache_name = kwargs.get(
            "sin_cache_name", self.sin_cache_global_name if self.window_size == -1 else self.sin_cache_local_name
        )
        return super().make_rotary_embedding_caches(cos_cache_name=cos_cache_name, sin_cache_name=sin_cache_name)

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        original_window_size = self.window_size
        original_head_size = self.head_size
        original_scale = self.attention_attrs["scale"]
        original_theta = self.rope_attrs["theta"]
        original_partial_rotary_factor = self.rope_attrs["partial_rotary_factor"]
        original_rotary_embedding_dim = self.rope_attrs["rotary_embedding_dim"]

        q_proj_out_features = (
            attention.q_proj.out_features if hasattr(attention, "q_proj") and hasattr(attention.q_proj, "out_features") else 0
        )
        is_full_attention = q_proj_out_features == self.num_attn_heads * self.global_head_dim

        if not is_full_attention:
            self.window_size = original_window_size
            self.head_size = self.local_head_dim
            self._set_rope_profile("sliding_attention")
        else:
            self.window_size = -1
            self.head_size = self.global_head_dim
            self._set_rope_profile("full_attention")

        # Match the source HF attention scaling per layer.
        if hasattr(attention, "scaling"):
            self.attention_attrs["scale"] = float(attention.scaling)

        # Gemma-4 K/V sharing is represented directly in loaded modules; builder
        # reads each layer's resolved projections without additional graph changes.
        _ = self.num_kv_shared_layers

        super().make_attention(layer_id, attention, root_input, **kwargs)

        self.window_size = original_window_size
        self.head_size = original_head_size
        self.attention_attrs["scale"] = original_scale
        self.rope_attrs["theta"] = original_theta
        self.rope_attrs["partial_rotary_factor"] = original_partial_rotary_factor
        self.rope_attrs["rotary_embedding_dim"] = original_rotary_embedding_dim

    def make_mlp(self, layer_id, mlp, root_input):
        original_intermediate_size = self.intermediate_size
        if self.use_double_wide_mlp:
            if hasattr(mlp, "gate_up_proj") and hasattr(mlp.gate_up_proj, "weight"):
                # Packed gate/up layout.
                self.intermediate_size = mlp.gate_up_proj.weight.shape[0] // 2
            elif hasattr(mlp, "gate_proj") and hasattr(mlp.gate_proj, "weight"):
                # Separate gate/up layout with per-layer width (e.g. 6144 or 12288).
                self.intermediate_size = mlp.gate_proj.weight.shape[0]
            elif hasattr(mlp, "up_proj") and hasattr(mlp.up_proj, "weight"):
                self.intermediate_size = mlp.up_proj.weight.shape[0]
        super().make_mlp(layer_id, mlp, root_input)
        self.intermediate_size = original_intermediate_size

    def make_model(self, input_path):
        # Make inputs and outputs to ONNX model
        self.make_inputs_and_outputs()
        self.make_preprocessing_nodes()

        model = self.load_weights(input_path)

        # Gemma-4 conditional models include vision/audio towers in the top-level
        # module tree. Export only the text decoder stack to avoid mixing towers.
        text_model = None
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            text_model = model.model.language_model
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            text_model = model.model
        elif hasattr(model, "layers"):
            text_model = model

        if text_model is None or not hasattr(text_model, "layers"):
            # Fallback to generic traversal if model layout is unexpected.
            return super().make_model(input_path)

        if not self.exclude_embeds and hasattr(text_model, "embed_tokens"):
            print("Reading embedding layer")
            self.make_embedding(text_model.embed_tokens.weight)
        else:
            self.layernorm_attrs["root_input"] = "inputs_embeds"
            self.layernorm_attrs["skip_input"] = "inputs_embeds"

        self.layer_id = 0
        for layer in text_model.layers:
            if self.layer_id >= self.num_layers:
                break
            print(f"Reading decoder layer {self.layer_id}")
            self.make_layer(self.layer_id, layer)
            self.layer_id += 1

        if hasattr(text_model, "norm"):
            print("Reading final norm")
            self.make_layernorm(
                self.layer_id,
                text_model.norm,
                skip=True,
                simple=self.layernorm_attrs["simple"],
                location="final_norm",
            )

        lm_head = model.lm_head if hasattr(model, "lm_head") else getattr(text_model, "lm_head", None)
        if lm_head is not None and not self.exclude_lm_head:
            print("Reading LM head")
            self.make_lm_head(lm_head)

        del model

    def make_key_value_cache_shape(self, layer_id, shape):
        # Gemma-4 mixes local/global attention head dimensions by layer.
        # Stamp the per-layer head size into past/present KV tensor shapes.
        kv_shape = list(super().make_key_value_cache_shape(layer_id, shape))
        if 0 <= layer_id < len(self.head_size_by_layer):
            kv_shape[3] = self.head_size_by_layer[layer_id]
        return kv_shape

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        super().make_genai_config(model_name_or_path, extra_kwargs, out_dir)
        genai_config_path = os.path.join(out_dir, "genai_config.json")
        with open(genai_config_path, encoding="utf-8") as f:
            genai_config = json.load(f)
        genai_config["model"]["decoder"]["head_size_by_layer"] = self.head_size_by_layer
        with open(genai_config_path, "w", encoding="utf-8") as f:
            json.dump(genai_config, f, indent=4)
