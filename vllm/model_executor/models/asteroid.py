# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Asteroid team.
# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Asteroid model compatible with HuggingFace weights."""
import copy
from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch import nn
from transformers import Qwen3Config

from vllm.attention import Attention, AttentionType
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .qwen3 import Qwen3Model
from .utils import AutoWeightsLoader, PPMissingLayer, maybe_prefix

logger = init_logger(__name__)


class AsteroidForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    """
    Asteroid model for causal language modeling with speech capabilities.
    """
    
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj", 
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config
        self.quant_config = quant_config

        # Get Asteroid-specific config parameters
        self.vocab_size = getattr(config, 'vocab_size', config.vocab_size)
        self.speech_vocab_sizes = getattr(config, 'speech_vocab_sizes', [1152] * 8)
        self.n_vq = getattr(config, 'n_vq', 8)
        self.group_size = getattr(config, 'group_size', 1)
        
        # Special token indices
        self.padding_idx = getattr(config, 'padding_idx', 151667) # <think> token in Qwen3 tokenizer
        self.sosp_idx = getattr(config, 'sosp_idx', 151646) # <|object_ref_start|> token in Qwen3 tokenizer
        self.eosp_idx = getattr(config, 'eosp_idx', 1024) # eosp token in speech vocab
        self.empty_idx = getattr(config, 'empty_idx', 151667) # the same as padding token
        
        # Calculate zero embedding indices
        self.zeroemb_idx = self.speech_vocab_sizes[0] - 1
        self.zeroemb_idx_list = [x - 1 for x in self.speech_vocab_sizes]
        
        # Text auxiliary loss inference mode
        self.text_auxiliary_loss_inference_mode = getattr(config, 'text_auxiliary_loss_inference_mode', True)

        # Main Qwen3 model
        self.model = Qwen3Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "model"))

        # Main LM head for text generation
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(config.vocab_size,
                                              config.hidden_size,
                                              quant_config=quant_config,
                                              prefix=maybe_prefix(prefix, "lm_head"))
        else:
            self.lm_head = PPMissingLayer()

        # Local transformer configuration for speech
        local_dim = config.hidden_size
        local_transformer_config = copy.deepcopy(config)
        local_transformer_config.hidden_size = local_dim
        local_transformer_config.num_hidden_layers = getattr(config, 'local_layers', 6)
        local_transformer_config.num_attention_heads = getattr(config, 'local_attn_heads', 16)
        local_transformer_config.num_key_value_heads = getattr(config, 'local_kv_heads', 16)
        local_transformer_config.intermediate_size = getattr(config, 'local_ffn_dim', 4096)
        local_transformer_config.attention_dropout = getattr(config, 'local_attn_dropout', 0.0)
        local_transformer_config.vocab_size = self.speech_vocab_sizes[0]
        local_transformer_config.pad_token_id = -100
        local_transformer_config.tie_word_embeddings = True
        self.local_transformer_config = local_transformer_config

        # Create local transformer for speech generation
        local_vllm_config = copy.deepcopy(vllm_config)
        local_vllm_config.model_config.hf_config = local_transformer_config
        if get_pp_group().is_last_rank:
            # only the last part needs the local transformer
            self.local_transformer = Qwen3Model(vllm_config=local_vllm_config,
                                                prefix=maybe_prefix(prefix, "local_transformer"))
            
            # Speech-specific components
            self.local_transformer_lm_heads = nn.ModuleList([
                nn.Linear(local_transformer_config.hidden_size,
                          self.speech_vocab_sizes[idx],
                          bias=False)
                for idx in range(self.n_vq)
            ])
        else:
            self.local_transformer = PPMissingLayer()
            self.local_transformer_lm_heads = PPMissingLayer()

        self.speech_embeddings = nn.ModuleList([
            nn.Embedding(self.speech_vocab_sizes[idx],
                        local_transformer_config.hidden_size,
                        padding_idx=self.zeroemb_idx_list[idx])
            for idx in range(self.n_vq)
        ])

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.speech_logits_processor = LogitsProcessor(self.speech_vocab_sizes[0])
        # Initialize tied weights for speech embeddings and heads
        self._tied_weights_keys = []
        for i in range(self.n_vq):
            self._tied_weights_keys.extend([
                f"speech_embeddings.{i}.weight",
                f"local_transformer_lm_heads.{i}.weight"
            ])

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """
        Forward pass for Asteroid model.
        During inference, this handles both text and speech token processing.
        """        
        if inputs_embeds is not None:
            # Use provided embeddings directly
            hidden_states = self.model(None, positions, intermediate_tensors, inputs_embeds)
        else:
            # Process input_ids to handle MIMO format
            B = input_ids.shape[0]
            
            # Reshape input_ids from flattened format to MIMO format
            if len(input_ids.shape) <= 2:
                # input_ids shape: [B, T * (n_vq + 1) * group_size]
                # Reshape to: [B, n_vq + 1, T * group_size]
                input_ids = input_ids.reshape(B, -1, self.n_vq + 1).transpose(1, 2).contiguous()
            
            input_ids = input_ids.int()
            
            # Extract text tokens (every group_size-th token from first channel)
            text_input_ids = input_ids[:, 0, ::self.group_size]
            
            # Handle empty tokens in text auxiliary loss mode
            if self.text_auxiliary_loss_inference_mode:
                mask = (text_input_ids != self.empty_idx).to(torch.int)
                text_input_ids = torch.where(
                    text_input_ids == self.empty_idx,
                    torch.tensor(0, dtype=text_input_ids.dtype, device=text_input_ids.device),
                    text_input_ids
                )
            else:
                mask = torch.ones_like(text_input_ids, dtype=torch.int)

            # Extract speech tokens
            speech_input_ids = input_ids[:, 1:, :].view(B, self.n_vq, -1, self.group_size).transpose(1, 2)

            # Create speech embeddings
            speech_embeddings = torch.zeros(
                B, speech_input_ids.size(1), self.group_size, self.local_transformer_config.hidden_size,
                device=input_ids.device, dtype=torch.float32
            )
            for i in range(self.n_vq):
                speech_embeddings += self.speech_embeddings[i](speech_input_ids[:, :, i, :])

            # Group speech embeddings (assuming group_size == 1 for simplicity)
            speech_grouped_embeddings = speech_embeddings.view(
                B, -1, self.group_size * self.local_transformer_config.hidden_size
            )

            # Get text embeddings
            text_embeds = self.model.get_input_embeddings(text_input_ids)

            # Combine text and speech embeddings based on mask
            inputs_embeds = (text_embeds * mask.unsqueeze(-1) + 
                           speech_grouped_embeddings * (1 - mask.unsqueeze(-1)))

            # Forward through main model
            hidden_states = self.model(None, positions, intermediate_tensors, inputs_embeds)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        """Compute logits based on current generation modality."""
        # Determine the current modality based on sampling metadata or sequence state
        current_modality = self._determine_current_modality(sampling_metadata)
        
        if current_modality == "text":
            # Standard text generation
            logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
            return logits
        elif current_modality == "speech":
            # Speech generation using local transformer
            return self._compute_speech_logits_with_local(hidden_states, sampling_metadata)
        else:
            # Fallback to text generation
            logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
            return logits
    
    def _determine_current_modality(self, sampling_metadata: SamplingMetadata) -> str:
        """Determine whether to generate text or speech tokens based on current state."""
        # This is a simplified implementation - you may need to adapt based on your specific logic
        # You could check the last generated tokens, special markers, or other state indicators
        
        # For now, we'll use a simple heuristic based on sequence groups
        for seq_group in sampling_metadata.seq_groups:
            for seq_data in seq_group.seq_data.values():
                # Check the last few tokens to determine modality
                if len(seq_data.output_token_ids) > 0:
                    # Use the existing get_token_modality method if available
                    # This would need to be adapted to work with the current sequence state
                    last_tokens = seq_data.output_token_ids[-10:]  # Check last 10 tokens
                    
                    # Check for speech start/end markers
                    if self.sosp_idx in last_tokens:
                        # Count SOSP and EOSP tokens to determine if we're in speech mode
                        sosp_count = last_tokens.count(self.sosp_idx)
                        eosp_count = last_tokens.count(self.eosp_idx)
                        if sosp_count > eosp_count:
                            return "speech"
        
        return "text"  # Default to text generation
    
    def _compute_speech_logits_with_local(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        """Compute speech logits using the forward_local method."""
        if not get_pp_group().is_last_rank:
            return None
            
        # Extract the last hidden state for local transformer input
        local_input = hidden_states[:, -1:, :]  # Take the last token's hidden state
        
        # Create dummy input_ids for speech (this might need adjustment based on your needs)
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        
        # For speech generation, we typically don't have input speech tokens for the next step
        # So we use empty input_ids
        speech_input_ids = torch.empty((batch_size, 0, self.n_vq), 
                                     dtype=torch.long, device=device)
        
        # Create a simple past_key_values mock (you may need to implement proper KV cache)
        class SimplePastKeyValues:
            def get_seq_length(self):
                return 0  # Simplified - you may need proper sequence length tracking
        
        past_key_values = SimplePastKeyValues()
        
        # Call forward_local to get speech logits
        local_logits_vq, _ = self.forward_local(
            local_last_hidden_states=local_input,
            input_ids=speech_input_ids,
            past_key_values=past_key_values
        )
        
        # Select which VQ head to use for current generation step
        if local_logits_vq and len(local_logits_vq) > 0:
            # You can implement more sophisticated VQ head selection logic here
            # For now, we use the first VQ head (index 0)
            selected_vq_idx = self._select_vq_head_for_generation(sampling_metadata)
            selected_logits = local_logits_vq[selected_vq_idx]
            
            # Create a speech-specific logits processor that accepts logits as input
            if not hasattr(self, '_speech_logits_processor_for_local'):
                from vllm.model_executor.layers.logits_processor import LogitsProcessor
                self._speech_logits_processor_for_local = LogitsProcessor(
                    vocab_size=selected_logits.shape[-1],  # Use actual vocab size from logits
                    logits_as_input=True,  # Important: tell processor we're passing logits directly
                    scale=1.0
                )
            
            # Apply speech logits processor
            speech_logits = self._speech_logits_processor_for_local(
                lm_head=None,  # No lm_head needed as logits are already computed
                hidden_states=selected_logits,  # Pass logits as hidden_states
                sampling_metadata=sampling_metadata
            )
            return speech_logits
        
        return None
    
    def _select_vq_head_for_generation(self, sampling_metadata: SamplingMetadata) -> int:
        """Select which VQ head to use for current generation step."""
        # This is a simplified implementation - you can implement more sophisticated logic
        # For example, you might want to:
        # 1. Track the current generation step for each sequence
        # 2. Use different VQ heads for different steps
        # 3. Use sequence-specific logic to determine the appropriate VQ head
        
        # For now, just return the first VQ head
        return 0

    def get_token_modality(self, seq: torch.Tensor) -> str:
        """
        Determine the modality of tokens to be generated ('speech' or 'text').
        """
        seq = seq.reshape(seq.shape[0], -1, (self.n_vq + 1), self.group_size).permute(0, 2, 3, 1)[0]
        text_channel = seq[0, 0, :]
        speech_group_first_channel = seq[1, 0, :]
        
        sosp_poses = torch.where(text_channel == self.sosp_idx)[0]
        eosp_poses = torch.where(speech_group_first_channel == self.eosp_idx)[0]
        
        if len(sosp_poses) == len(eosp_poses):
            return "text"
        return "speech"

    def forward_local(self,
                      local_last_hidden_states: torch.Tensor,
                      input_ids: torch.Tensor,
                      past_key_values: Optional[dict] = None,
                      **kwargs):
        """
        Forward pass for local transformer (speech generation).
        """
        cached_len = past_key_values.get_seq_length() if past_key_values is not None else 0
        
        input_embs = local_last_hidden_states
        if input_ids.numel():
            speech_embeddings = torch.zeros(
                input_ids.size(0), 0, local_last_hidden_states.size(-1),
                dtype=local_last_hidden_states.dtype,
                device=local_last_hidden_states.device
            )
            for i in range(self.n_vq):
                speech_embeddings += self.speech_embeddings[i](input_ids[:, :, i])
            input_embs = torch.cat([local_last_hidden_states, speech_embeddings], dim=1)
        
        # Keep only new tokens
        input_embs = input_embs[:, cached_len:, :]
        
        # Create positions tensor
        positions = torch.arange(cached_len, cached_len + input_embs.size(1), 
                               device=input_embs.device, dtype=torch.long)
        
        # Forward through local transformer
        output = self.local_transformer(None, positions, None, input_embs)
        local_last_hidden_states = output[:, -1, :]
        
        # Generate logits for each VQ head
        local_logits_vq = [lm_head(local_last_hidden_states) for lm_head in self.local_transformer_lm_heads]
        
        return local_logits_vq, past_key_values

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load model weights with support for Asteroid-specific components."""
        skip_prefixes = []
        
        # Skip main lm_head if tied with embeddings
        if self.config.tie_word_embeddings:
            skip_prefixes.append("lm_head.")
        
        # Skip local transformer lm heads if tied with speech embeddings
        # The weights are tied as indicated by self._tied_weights_keys
        for i in range(self.n_vq):
            skip_prefixes.append(f"local_transformer_lm_heads.{i}.")
        
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=skip_prefixes if skip_prefixes else None,
        )
        return loader.load_weights(weights)