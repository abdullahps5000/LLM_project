"""
Lightweight stage model that loads only needed layers from safetensors.
Uses transformers library but loads weights selectively.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

from .errors import ModelError
from .logging_config import get_logger

logger = get_logger("ebp.stage_model")


class StageModel:
    """
    Model for a single pipeline stage.
    Loads only the layers assigned to this stage from safetensors.
    """
    
    def __init__(
        self,
        model_path: str,
        stage_path: str,
        layer_range: Tuple[int, int],
        stage_id: str,
        is_first_stage: bool = False,
        is_last_stage: bool = False,
    ):
        self.model_path = model_path
        self.stage_path = stage_path
        self.layer_range = layer_range
        self.stage_id = stage_id
        self.is_first_stage = is_first_stage
        self.is_last_stage = is_last_stage
        
        # Load config
        try:
            self.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        except Exception as e:
            raise ModelError(f"Failed to load model config: {e}", model_path=model_path)
        
        # Load full model structure (lightweight - just structure, not weights)
        # We'll replace weights with stage-specific ones
        # IMPORTANT: Load with low_cpu_mem_usage to avoid loading all weights
        try:
            logger.info(f"Loading model structure for stage {stage_id}...")
            # Try to load with low_cpu_mem_usage first
            # If model files don't exist (e.g., on remote agents), use from_config
            if os.path.exists(model_path) and os.path.exists(os.path.join(model_path, "config.json")):
                # Model directory exists, try loading normally
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        local_files_only=True,
                        dtype=torch.float16,
                        low_cpu_mem_usage=True,  # Don't load all weights into memory
                    )
                except Exception as e:
                    # If loading fails (e.g., missing weight files), try from_config
                    logger.warning(f"Failed to load model with weights, trying from_config: {e}")
                    self.model = AutoModelForCausalLM.from_config(
                        self.config,
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                    )
            else:
                # Model directory doesn't exist, create from config only
                logger.info(f"Model directory not found at {model_path}, creating structure from config only")
                self.model = AutoModelForCausalLM.from_config(
                    self.config,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                )
            # Move to CPU explicitly (no device_map needed)
            self.model = self.model.to("cpu")
        except Exception as e:
            raise ModelError(f"Failed to load model structure: {e}", model_path=model_path)
        
        # Extract only the layers we need
        self._extract_stage_layers()
        
        # Load weights from stage safetensors
        self._load_stage_weights()
        
        # Set to eval mode
        self.model.eval()
        
        logger.info(f"Stage {stage_id} model loaded: layers {layer_range[0]}-{layer_range[1]}")
    
    def _extract_stage_layers(self):
        """Extract only the layers needed for this stage."""
        # Get all layers
        all_layers = self.model.model.layers
        
        # Extract only our layer range
        start_idx = self.layer_range[0]
        end_idx = self.layer_range[1] + 1  # +1 because range is inclusive
        
        # Create a new module list with only our layers
        self.stage_layers = all_layers[start_idx:end_idx]
        
        # IMPORTANT: Keep the full model structure for Qwen2 rotary embeddings
        # Qwen2 layers need access to the model's rotary embedding computation
        # So we keep all layers in the model but only use our subset in forward()
        
        # For first stage: keep embeddings
        if not self.is_first_stage:
            # Remove embeddings - we'll receive hidden states
            self.model.model.embed_tokens = None
        
        # For last stage: keep LM head
        if not self.is_last_stage:
            # Remove LM head - we'll output hidden states
            self.model.lm_head = None
        
        # Keep all layers in model (needed for rotary embeddings)
        # But we'll only use our subset in forward()
        # Don't replace: self.model.model.layers = self.stage_layers
        
        logger.debug(f"Extracted {len(self.stage_layers)} layers ({self.layer_range[0]}-{self.layer_range[1]})")
    
    def _load_stage_weights(self):
        """Load weights from stage safetensors file."""
        if not os.path.exists(self.stage_path):
            raise ModelError(f"Stage file not found: {self.stage_path}", model_path=self.stage_path)
        
        logger.info(f"Loading weights from {self.stage_path}...")
        
        # Load state dict from safetensors
        state_dict = {}
        with safe_open(self.stage_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                state_dict[key] = tensor
        
        # Load into model (only matching keys)
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys (will use default): {len(missing_keys)} keys")
        if unexpected_keys:
            logger.warning(f"Unexpected keys (ignored): {len(unexpected_keys)} keys")
        
        logger.info(f"Loaded {len(state_dict)} tensors from stage file")
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: bool = True,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through stage.
        
        Args:
            input_ids: Token IDs (first stage only)
            hidden_states: Hidden states from previous stage (not first stage)
            attention_mask: Attention mask
            past_key_values: Cached key-value pairs for incremental decoding
            use_cache: Whether to return KV cache
            position_ids: Position IDs (if None, computed from sequence length)
        
        Returns:
            Hidden states (not last stage) or logits (last stage)
            If use_cache=True, returns tuple (output, past_key_values)
        """
        with torch.no_grad():
            # Get model dtype
            model_dtype = next(self.model.parameters()).dtype if list(self.model.parameters()) else torch.float16
            
            # First stage: process input_ids
            if self.is_first_stage:
                if input_ids is None:
                    raise ValueError("First stage requires input_ids")
                
                # Get embeddings
                if self.model.model.embed_tokens is not None:
                    hidden_states = self.model.model.embed_tokens(input_ids)
                    # Ensure correct dtype
                    hidden_states = hidden_states.to(dtype=model_dtype)
                else:
                    raise RuntimeError("Embeddings not available in first stage")
            else:
                # Not first stage: ensure hidden_states have correct dtype
                if hidden_states is not None:
                    hidden_states = hidden_states.to(dtype=model_dtype)
            
            # Compute position_ids for rotary embeddings (Qwen2 needs this)
            # Position IDs are just [0, 1, 2, ..., seq_len-1]
            batch_size, seq_len = hidden_states.shape[:2]
            if attention_mask is not None:
                # Use attention mask to determine actual sequence lengths
                position_ids = attention_mask.long().cumsum(dim=-1) - 1
                position_ids = position_ids.clamp(min=0)
            else:
                # Create position_ids from sequence length
                position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device)
                position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
            # Compute rotary position embeddings (cos, sin) from position_ids
            # Qwen2Model.forward computes position_embeddings once using self.rotary_emb(hidden_states, position_ids)
            # and passes it to all decoder layers. We need to do the same.
            # The rotary_emb module is in model.model.rotary_emb
            if hasattr(self.model.model, "rotary_emb") and self.model.model.rotary_emb is not None:
                # rotary_emb.forward(hidden_states, position_ids) returns (cos, sin) tuple
                # This is the same computation done in Qwen2Model.forward
                position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)
            else:
                raise RuntimeError(
                    "rotary_emb not found in model.model. "
                    "This is required for Qwen2 models. "
                    "Make sure the full model structure is preserved."
                )
            
            # Process through transformer layers with KV cache support
            # Qwen2DecoderLayer.forward accepts position_embeddings (not position_ids)
            # We pass the pre-computed position_embeddings to each layer
            new_past_key_values = [] if use_cache else None
            
            for i, layer in enumerate(self.stage_layers):
                layer_idx = self.layer_range[0] + i
                
                # Get past_key_values for this layer if available
                layer_past = None
                if past_key_values is not None and i < len(past_key_values):
                    layer_past = past_key_values[i]
                
                # Qwen2DecoderLayer.forward signature:
                # forward(hidden_states, attention_mask=None, position_ids=None, position_embeddings=None, 
                #         past_key_values=None, use_cache=False, ...)
                # We pass position_embeddings which is the (cos, sin) tuple computed above
                
                try:
                    layer_output = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,  # Some models might still need this
                        position_embeddings=position_embeddings,  # This is what Qwen2Attention actually needs
                        past_key_values=layer_past,
                        use_cache=use_cache,
                    )
                except Exception as e:
                    logger.error(f"Layer {layer_idx} forward failed: {e}")
                    raise RuntimeError(
                        f"Layer {layer_idx} forward pass failed. "
                        f"Error: {e}. "
                        f"Make sure position_embeddings are correctly computed."
                    ) from e
                
                # Handle output format - Qwen2 layers return (hidden_states, past_key_values) tuple when use_cache=True
                if isinstance(layer_output, tuple):
                    hidden_states = layer_output[0]  # Extract hidden_states from tuple
                    if use_cache and len(layer_output) > 1:
                        # Store past_key_values for this layer
                        new_past_key_values.append(layer_output[1])
                elif isinstance(layer_output, torch.Tensor):
                    hidden_states = layer_output  # Direct tensor output
                    if use_cache:
                        # No KV cache returned, append None
                        new_past_key_values.append(None)
                else:
                    raise RuntimeError(f"Unexpected layer output type: {type(layer_output)}")
            
            # Last stage: apply LM head
            if self.is_last_stage:
                if self.model.lm_head is not None:
                    # Apply final layer norm if exists
                    if hasattr(self.model.model, "norm") and self.model.model.norm is not None:
                        hidden_states = self.model.model.norm(hidden_states)
                    logits = self.model.lm_head(hidden_states)
                    if use_cache:
                        return logits, new_past_key_values
                    return logits
            
            if use_cache:
                return hidden_states, new_past_key_values
            return hidden_states


def load_stage_model(
    model_path: str,
    stage_path: str,
    layer_range: Tuple[int, int],
    stage_id: str,
    is_first_stage: bool = False,
    is_last_stage: bool = False,
) -> StageModel:
    """
    Load a stage model from safetensors.
    
    Args:
        model_path: Path to original model directory
        stage_path: Path to stage safetensors file
        layer_range: (start_layer, end_layer) inclusive
        stage_id: Stage identifier
        is_first_stage: Whether this is the first stage
        is_last_stage: Whether this is the last stage
    
    Returns:
        Loaded StageModel
    """
    return StageModel(
        model_path=model_path,
        stage_path=stage_path,
        layer_range=layer_range,
        stage_id=stage_id,
        is_first_stage=is_first_stage,
        is_last_stage=is_last_stage,
    )

