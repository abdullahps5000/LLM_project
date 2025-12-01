"""
Enhanced inference engine with KV cache, sessions, binary protocol, and progress tracking.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np
import torch
from transformers import AutoTokenizer

from .config import get_config
from .errors import AgentError, ModelError, NetworkError
from .logging_config import get_logger
from .retry import http_retry
from .serialization import deserialize_tensor, serialize_tensor

logger = get_logger("ebp.inference_engine")


class DistributedInferenceEngine:
    """
    Enhanced inference engine with KV cache support for incremental decoding.
    """
    
    def __init__(
        self,
        plan_path: str,
        agent_urls: Optional[Dict[str, str]] = None,
        auto_load_stages: bool = True,
        use_binary: bool = True,  # Use binary protocol by default
        auth_token: Optional[str] = None,
    ):
        """
        Initialize inference engine from plan.json.
        
        Args:
            plan_path: Path to plan.json
            agent_urls: Optional mapping of agent names to URLs
            auto_load_stages: Automatically load stages on initialization
            use_binary: Use binary protocol for tensor transfer (faster)
            auth_token: Authentication token for agents
        """
        self.plan_path = os.path.abspath(plan_path)
        with open(self.plan_path, "r") as f:
            self.plan = json.load(f)
        
        self.model_path = self.plan["model_path"]
        self.model_name = self.plan["model_name"]
        self.pipeline_order = self.plan["pipeline_order"]
        self.layer_ranges = self.plan["layer_ranges"]
        self.use_binary = use_binary
        self.auth_token = auth_token or os.environ.get("EBP_AUTH_TOKEN", "")
        
        # Agent URLs
        if agent_urls:
            self.agent_urls = agent_urls
        else:
            self.agent_urls = {
                "pc": "http://127.0.0.1:8008",
                "pi": "http://172.20.10.2:8008",
            }
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Tokenizer loaded from {self.model_path}")
        except Exception as e:
            raise ModelError(f"Failed to load tokenizer: {e}", model_path=self.model_path)
        
        self.config = get_config()
        logger.info(f"Inference engine initialized: {len(self.pipeline_order)} stages")
        
        # Session IDs per stage (for KV cache)
        self.session_ids: Dict[int, str] = {}
        
        # Auto-load stages if requested
        if auto_load_stages:
            self._load_all_stages()
    
    def _load_all_stages(self) -> None:
        """Automatically load all stages on agents."""
        import glob
        
        plan_dir = os.path.dirname(self.plan_path)
        stages_dir = os.path.join(plan_dir, "stages_out")
        
        if not os.path.exists(stages_dir):
            stages_dir = os.path.join(plan_dir, "..", "stages_out")
            stages_dir = os.path.abspath(stages_dir)
        
        if not os.path.exists(stages_dir):
            logger.warning("stages_out directory not found, stages must be loaded manually")
            return
        
        pattern = os.path.join(stages_dir, f"{self.model_name}_*")
        dirs = sorted(glob.glob(pattern), reverse=True)
        
        if not dirs:
            logger.warning("No stage directories found, stages must be loaded manually")
            return
        
        latest_stage_dir = dirs[0]
        logger.info(f"Auto-loading stages from {os.path.basename(latest_stage_dir)}")
        
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception:
            local_ip = "127.0.0.1"
        
        latest_dir = os.path.basename(latest_stage_dir)
        file_server_url = f"http://{local_ip}:8090/{latest_dir}"
        
        for i, name in enumerate(self.pipeline_order):
            stage_id = f"{name}-stage{i}"
            agent_url = self.agent_urls.get(name)
            layer_range = tuple(self.layer_ranges.get(name, [0, 0]))
            is_first_stage = (i == 0)
            is_last_stage = (i == len(self.pipeline_order) - 1)
            
            if not agent_url:
                logger.warning(f"Unknown agent {name}, skipping stage load")
                continue
            
            stage_file = f"stage_{i}.safetensors"
            stage_path = os.path.join(latest_stage_dir, stage_file)
            
            if name == "pc" and os.path.exists(stage_path):
                stage_url = f"file://{stage_path}"
            else:
                stage_url = f"{file_server_url}/{stage_file}"
            
            logger.info(f"Loading stage {i} ({name})...")
            self._load_stage(agent_url, stage_id, stage_url, layer_range, is_first_stage, is_last_stage)
    
    @http_retry(max_retries=3, initial_delay=1.0, max_delay=60.0)
    def _load_stage(
        self,
        agent_url: str,
        stage_id: str,
        stage_url: str,
        layer_range: tuple,
        is_first_stage: bool,
        is_last_stage: bool,
    ) -> bool:
        """Load a stage on an agent."""
        try:
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            
            with httpx.Client(timeout=600.0, trust_env=False) as client:
                r = client.post(
                    agent_url.rstrip("/") + "/v1/stage/load",
                    json={
                        "stage_id": stage_id,
                        "stage_url": stage_url,
                        "timeout_s": 600.0,
                        "mode": "metadata",
                        "model_path": self.model_path,
                        "layer_range": list(layer_range),
                        "is_first_stage": is_first_stage,
                        "is_last_stage": is_last_stage,
                    },
                    headers=headers,
                )
                r.raise_for_status()
                resp = r.json()
                
                if not resp.get("ok", False):
                    logger.error(f"Failed to load stage: {resp.get('error', 'unknown error')}")
                    return False
                
                cached = resp.get("cached", False)
                status = "cached" if cached else "loaded"
                logger.info(f"  ✓ {status.capitalize()} {stage_id}: {resp.get('tensor_count', 0)} tensors")
                return True
        except Exception as e:
            logger.error(f"Error loading stage {stage_id}: {e}")
            return False
    
    def _start_sessions(self, max_length: int = 2048) -> None:
        """Start sessions on all agents for KV cache."""
        for i, name in enumerate(self.pipeline_order):
            stage_id = f"{name}-stage{i}"
            agent_url = self.agent_urls.get(name)
            
            if not agent_url:
                continue
            
            try:
                headers = {}
                if self.auth_token:
                    headers["Authorization"] = f"Bearer {self.auth_token}"
                
                with httpx.Client(timeout=10.0, trust_env=False) as client:
                    r = client.post(
                        agent_url.rstrip("/") + "/v1/session/start",
                        json={"stage_id": stage_id, "max_length": max_length},
                        headers=headers,
                    )
                    r.raise_for_status()
                    resp = r.json()
                    
                    if resp.get("ok", False):
                        session_id = resp.get("session_id")
                        self.session_ids[i] = session_id
                        logger.debug(f"Started session {session_id} for stage {i}")
            except Exception as e:
                logger.warning(f"Failed to start session for stage {i}: {e}")
    
    def _end_sessions(self) -> None:
        """End all sessions."""
        for i, session_id in self.session_ids.items():
            agent_name = self.pipeline_order[i]
            agent_url = self.agent_urls.get(agent_name)
            
            if not agent_url:
                continue
            
            try:
                headers = {}
                if self.auth_token:
                    headers["Authorization"] = f"Bearer {self.auth_token}"
                
                with httpx.Client(timeout=5.0, trust_env=False) as client:
                    r = client.post(
                        agent_url.rstrip("/") + "/v1/session/end",
                        json={"session_id": session_id},
                        headers=headers,
                    )
                    r.raise_for_status()
            except Exception as e:
                logger.debug(f"Failed to end session {session_id}: {e}")
        
        self.session_ids.clear()
    
    @http_retry(max_retries=3, initial_delay=1.0, max_delay=60.0)
    def _forward_stage(
        self,
        stage_idx: int,
        input_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        is_prefill: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through a single stage with KV cache support.
        
        Args:
            stage_idx: Stage index in pipeline
            input_ids: Token IDs (first stage only, for prefill: full sequence, for decode: single token)
            hidden_states: Hidden states (not first stage, for prefill: full sequence, for decode: single token)
            is_prefill: Whether this is the prefill phase (full sequence) or decode phase (single token)
        """
        agent_name = self.pipeline_order[stage_idx]
        agent_url = self.agent_urls.get(agent_name)
        stage_id = f"{agent_name}-stage{stage_idx}"
        session_id = self.session_ids.get(stage_idx)
        
        if not agent_url:
            raise AgentError(f"Unknown agent: {agent_name}", agent_url=None)
        
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        timeout = max(120.0, self.config.network.timeout_s * 20)
        
        if self.use_binary:
            # Binary protocol
            if stage_idx == 0:
                if input_ids is None:
                    raise ValueError("First stage requires input_ids")
                
                # Serialize tensor
                input_bytes = serialize_tensor(input_ids)
                
                headers.update({
                    "Content-Type": "application/octet-stream",
                    "X-Stage-Id": stage_id,
                    "X-Session-Id": session_id or "",
                    "X-Input-Type": "input_ids" if is_prefill else "input_ids_new",
                    "X-Batch-Size": str(input_ids.shape[0]),
                    "X-Seq-Len": str(input_ids.shape[1]),
                })
                
                with httpx.Client(timeout=timeout, trust_env=False) as client:
                    r = client.post(
                        agent_url.rstrip("/") + "/v1/inference/forward",
                        content=input_bytes,
                        headers=headers,
                    )
                    r.raise_for_status()
                    
                    if r.status_code != 200:
                        error = r.headers.get("X-Error", "Unknown error")
                        raise AgentError(f"Forward pass failed on {agent_name}: {error}", agent_url=agent_url)
                    
                    output = deserialize_tensor(r.content)
                    return output
            else:
                if hidden_states is None:
                    raise ValueError(f"Stage {stage_idx} requires hidden_states")
                
                hidden_bytes = serialize_tensor(hidden_states)
                
                headers.update({
                    "Content-Type": "application/octet-stream",
                    "X-Stage-Id": stage_id,
                    "X-Session-Id": session_id or "",
                    "X-Input-Type": "hidden_states" if is_prefill else "hidden_states_new",
                    "X-Batch-Size": str(hidden_states.shape[0]),
                    "X-Seq-Len": str(hidden_states.shape[1]),
                    "X-Hidden-Size": str(hidden_states.shape[2]),
                })
                
                with httpx.Client(timeout=timeout, trust_env=False) as client:
                    r = client.post(
                        agent_url.rstrip("/") + "/v1/inference/forward",
                        content=hidden_bytes,
                        headers=headers,
                    )
                    r.raise_for_status()
                    
                    if r.status_code != 200:
                        error = r.headers.get("X-Error", "Unknown error")
                        raise AgentError(f"Forward pass failed on {agent_name}: {error}", agent_url=agent_url)
                    
                    output = deserialize_tensor(r.content)
                    return output
        else:
            # JSON protocol (backward compatible)
            if stage_idx == 0:
                if input_ids is None:
                    raise ValueError("First stage requires input_ids")
                
                batch_size, seq_len = input_ids.shape
                
                if is_prefill:
                    input_ids_list = input_ids.cpu().numpy().flatten().tolist()
                    request_data = {
                        "stage_id": stage_id,
                        "session_id": session_id,
                        "input_ids": input_ids_list,
                        "input_ids_new": None,
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                    }
                else:
                    # Single token
                    input_ids_list = input_ids[0, -1:].cpu().numpy().flatten().tolist()
                    request_data = {
                        "stage_id": stage_id,
                        "session_id": session_id,
                        "input_ids": None,
                        "input_ids_new": input_ids_list,
                        "batch_size": batch_size,
                        "seq_len": 1,
                    }
            else:
                if hidden_states is None:
                    raise ValueError(f"Stage {stage_idx} requires hidden_states")
                
                batch_size, seq_len, hidden_size = hidden_states.shape
                
                if is_prefill:
                    hidden_list = hidden_states.cpu().numpy().flatten().tolist()
                    request_data = {
                        "stage_id": stage_id,
                        "session_id": session_id,
                        "hidden_states": hidden_list,
                        "hidden_states_new": None,
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "hidden_size": hidden_size,
                    }
                else:
                    # Single token
                    hidden_list = hidden_states[0, -1:].cpu().numpy().flatten().tolist()
                    request_data = {
                        "stage_id": stage_id,
                        "session_id": session_id,
                        "hidden_states": None,
                        "hidden_states_new": hidden_list,
                        "batch_size": batch_size,
                        "seq_len": 1,
                        "hidden_size": hidden_size,
                    }
            
            with httpx.Client(timeout=timeout, trust_env=False) as client:
                r = client.post(
                    agent_url.rstrip("/") + "/v1/inference/forward",
                    json=request_data,
                    headers=headers,
                )
                r.raise_for_status()
                resp = r.json()
                
                if not resp.get("ok", False):
                    error_msg = resp.get('error', 'unknown error')
                    raise AgentError(
                        f"Forward pass failed on {agent_name}: {error_msg}",
                        agent_url=agent_url,
                    )
                
                if "output" not in resp or "output_shape" not in resp:
                    raise AgentError(
                        f"Invalid response from {agent_name}: missing 'output' or 'output_shape'",
                        agent_url=agent_url,
                    )
                
                output_list = resp["output"]
                output_shape = resp["output_shape"]
                output_array = np.array(output_list, dtype=np.float32).reshape(output_shape)
                output = torch.from_numpy(output_array)
                return output
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        stream: bool = False,
    ) -> str:
        """
        Generate text using the distributed pipeline with KV cache.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            stream: Stream tokens as they're generated
        
        Returns:
            Generated text
        """
        logger.info(f"Generating {max_new_tokens} tokens for prompt: '{prompt[:100]}...'")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=False)
        input_ids = inputs["input_ids"]
        
        logger.debug(f"Input tokens: {input_ids.shape}")
        
        # Start sessions for KV cache
        self._start_sessions(max_length=input_ids.shape[1] + max_new_tokens)
        
        try:
            # Progress bar
            try:
                from tqdm import tqdm
                pbar = tqdm(total=max_new_tokens, desc="Generating", unit="token")
                use_tqdm = True
            except ImportError:
                pbar = None
                use_tqdm = False
                logger.info("tqdm not available, progress will be logged instead")
            
            start_time = time.time()
            stage_times = []
            
            # Prefill: process full prompt once
            logger.info("Prefill: processing prompt...")
            prefill_start = time.perf_counter()
            
            current_hidden = None
            for stage_idx in range(len(self.pipeline_order)):
                stage_start = time.perf_counter()
                
                if stage_idx == 0:
                    current_hidden = self._forward_stage(stage_idx, input_ids=input_ids, is_prefill=True)
                else:
                    current_hidden = self._forward_stage(stage_idx, hidden_states=current_hidden, is_prefill=True)
                
                stage_time = (time.perf_counter() - stage_start) * 1000
                stage_times.append((self.pipeline_order[stage_idx], stage_time))
            
            prefill_time = (time.perf_counter() - prefill_start) * 1000
            logger.info(f"Prefill complete: {prefill_time:.1f}ms")
            
            # Get logits for last token of prompt
            logits = current_hidden
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature and sample first token
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            if do_sample:
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            else:
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True).unsqueeze(0)
            
            generated_ids = torch.cat([input_ids, next_token_id], dim=1)
            
            # Decode first token
            new_token_text = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            if stream:
                print(new_token_text, end='', flush=True)
            
            if use_tqdm and pbar:
                pbar.update(1)
                pbar.set_postfix({'token': new_token_text[:20]})
            
            # Decode loop: generate remaining tokens one at a time (with KV cache)
            for step in range(1, max_new_tokens):
                step_start = time.perf_counter()
                
                # Forward pass with single new token (KV cache used)
                current_hidden = None
                new_token_ids = next_token_id  # [1, 1]
                
                for stage_idx in range(len(self.pipeline_order)):
                    stage_start = time.perf_counter()
                    
                    if stage_idx == 0:
                        current_hidden = self._forward_stage(stage_idx, input_ids=new_token_ids, is_prefill=False)
                    else:
                        current_hidden = self._forward_stage(stage_idx, hidden_states=current_hidden, is_prefill=False)
                    
                    stage_time = (time.perf_counter() - stage_start) * 1000
                    stage_times.append((self.pipeline_order[stage_idx], stage_time))
                
                # Get logits for new token
                logits = current_hidden
                next_token_logits = logits[0, -1, :]
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Sample next token
                if do_sample:
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).unsqueeze(0)
                else:
                    next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True).unsqueeze(0)
                
                # Append token
                generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
                
                # Decode and display
                new_token_text = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
                step_time = (time.perf_counter() - step_start) * 1000
                tokens_per_sec = 1000.0 / step_time if step_time > 0 else 0
                
                if stream:
                    print(new_token_text, end='', flush=True)
                
                if use_tqdm and pbar:
                    pbar.set_postfix({
                        'token': new_token_text[:20],
                        'speed': f'{tokens_per_sec:.2f} tok/s'
                    })
                    pbar.update(1)
                else:
                    if (step + 1) % 5 == 0 or step == 0:
                        logger.info(f"Step {step + 1}/{max_new_tokens}: '{new_token_text}' ({tokens_per_sec:.2f} tok/s)")
                
                # Check EOS
                if next_token_id.item() == self.tokenizer.eos_token_id:
                    logger.info(f"Generated EOS token at step {step + 1}")
                    break
            
            if use_tqdm and pbar:
                pbar.close()
            
            # Timing summary
            total_time = time.time() - start_time
            total_tokens = generated_ids.shape[1] - input_ids.shape[1]
            avg_tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
            
            logger.info(f"Generation complete: {total_tokens} tokens in {total_time:.2f}s ({avg_tokens_per_sec:.2f} tok/s)")
            
            # Per-stage timing
            if stage_times:
                from collections import defaultdict
                stage_stats = defaultdict(list)
                for agent, time_taken in stage_times:
                    stage_stats[agent].append(time_taken)
                
                logger.info("Per-stage average timing:")
                for agent, times in stage_stats.items():
                    if times:
                        avg_time = sum(times) / len(times)
                        logger.info(f"  {agent}: {avg_time:.1f}ms avg")
            
            # Decode final text
            generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            if stream:
                print()  # Newline after streaming
            
            return generated_text
        
        finally:
            # Always end sessions
            self._end_sessions()
