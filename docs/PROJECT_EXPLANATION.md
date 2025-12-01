# EBP (Edge-Based Pipeline) - Complete Project Explanation

## 🎯 Project Goal

Build a **scalable distributed LLM inference system** that splits large transformer models across multiple edge devices (PC, Raspberry Pi, phones) using **pipeline parallelism**. This allows running models that are too large for any single device.

---

## 🏗️ Architecture Overview

### High-Level Flow

```
┌─────────────┐
│  Coordinator │  (Runs on PC)
│              │  1. Discovers agents
│              │  2. Profiles compute/memory
│              │  3. Partitions model layers
│              │  4. Packages stage files
│              │  5. Serves files via HTTP
│              │  6. Instructs agents to load stages
└──────┬───────┘
       │
       ├─────────────────┐
       │                 │
┌──────▼──────┐   ┌──────▼──────┐
│  PC Agent   │   │  Pi Agent   │
│             │   │             │
│ FastAPI     │   │ FastAPI     │
│ Server      │   │ Server      │
│ Port 8008   │   │ Port 8008   │
│             │   │             │
│ Stage 0:    │   │ Stage 1:    │
│ Layers 0-14 │   │ Layers 15-27│
└─────────────┘   └─────────────┘
```

---

## 📦 Components Breakdown

### 1. **Agent Service** (`ebp/agent_main.py` + `ebp/agent_app.py`)

**What it does:**
- Runs a FastAPI web server on each device (PC, Pi, etc.)
- Exposes REST API endpoints for discovery, profiling, and stage loading
- Advertises itself on the network (optional mDNS)

**Key Endpoints:**

| Endpoint | Purpose |
|----------|---------|
| `/v1/health` | Health check - returns agent ID, name, timestamp |
| `/v1/capabilities` | Device specs - CPU count, RAM (total/available), platform info |
| `/v1/profile/matmul` | Compute profiling - runs matrix multiplication benchmark to estimate GFLOPS |
| `/v1/net/ping` | Network latency - measures RTT to other agents |
| `/v1/stage/load` | **Stage loading** - downloads and loads model stage file (metadata-only by default) |

**How it works:**
```python
# Agent runs continuously, waiting for coordinator requests
python -m ebp.agent_main --name pi --port 8008
```

**Memory Safety:**
- Stage loading uses `metadata_only=True` by default
- Reads safetensors metadata (keys, shapes, dtypes) without loading tensors into RAM
- Prevents OOM on low-memory devices

---

### 2. **Coordinator** (`ebp/coordinator_main.py`)

**What it does:**
The orchestrator that coordinates the entire distributed inference setup.

**Step-by-Step Process:**

#### Step 1: Discovery & Profiling
```python
# Queries each agent for capabilities
caps = call_capabilities(url)  # Gets CPU, RAM info
prof = call_profile(url)        # Gets GFLOPS estimate
```

**Output:**
```
[coordinator] Agents:
  - pc @ http://127.0.0.1:8008 | CPU=2 | RAM=3.75GB | eff_GFLOPS=35.80
  - pi @ http://172.20.10.2:8008 | CPU=4 | RAM=7.05GB | eff_GFLOPS=30.71
```

#### Step 2: Model Analysis
```python
# Reads model config (no weight loading)
L, H = model_shape(model_path)  # Gets num_layers, hidden_size
layer_bytes = estimate_layer_bytes(model_path, L)  # Estimates memory per layer
```

**How layer memory estimation works:**
- Opens safetensors files in metadata-only mode
- Reads tensor shapes and dtypes
- Calculates: `bytes = num_elements × bytes_per_element`
- Groups by layer using regex patterns (e.g., `model.layers.0.*`)

#### Step 3: Memory Budget Calculation
```python
# Conservative memory budgets (reserves space for KV cache + OS)
budget = available_ram × mem_fraction  # Default: 0.45-0.60
```

**Example:**
- PC: 3.75GB available → 1.69GB budget (45%)
- Pi: 7.05GB available → 3.17GB budget (45%)

#### Step 4: Dynamic Programming Partitioning
```python
layer_ranges = dp_partition_layers(
    layer_costs=[1.0] * 28,           # Compute cost per layer
    device_gflops=[35.80, 30.71],     # Device speeds
    layer_bytes=[110MB] * 28,          # Memory per layer
    device_mem_budget_bytes=[1.69GB, 3.17GB],  # Hard memory limits
    min_prefix=4                       # First 4 layers on first device
)
```

**Algorithm:**
- Uses binary search to find optimal bottleneck time
- Enforces **hard memory constraints** (stage_bytes ≤ budget)
- Minimizes max stage time (bottleneck)
- Returns contiguous layer ranges: `[(0, 14), (15, 27)]`

**Output:**
```
=== DP Partition (memory-constrained) ===
  pc: layers [0..14] (15 layers) | mem=1.54GB/1.69GB (91.4%)
  pi: layers [15..27] (13 layers) | mem=1.34GB/3.17GB (42.1%)
```

#### Step 5: Stage Packaging (if `--package` flag)
```python
stage_paths = package_stages(
    model_dir=model_path,
    out_dir="stages_out/run_id",
    layer_ranges=[(0,14), (15,27)],
    dtype="fp16"
)
```

**How packaging works:**
1. **Streaming approach**: Processes one stage at a time
2. **Batch loading**: Loads tensors in small batches (5-20 at a time)
3. **Memory efficient**: 
   - Collects keys first (metadata-only)
   - Loads tensors in batches
   - Writes stage file immediately
   - Clears memory before next stage
4. **Tensor assignment**:
   - Layer tensors: `model.layers.N.*` → assigned to stage containing layer N
   - Global tensors: embeddings → first stage, LM head → last stage

**Output files:**
```
stages_out/Qwen2.5-1.5B-Instruct_1764404816/
├── stage_0.safetensors  (PC: layers 0-14)
├── stage_1.safetensors  (Pi: layers 15-27)
├── stage_0.meta.json
├── stage_1.meta.json
└── manifest.json
```

#### Step 6: HTTP File Server
```python
# Serves stage files so agents can download them
srv = serve_directory(parent_dir="stages_out", port=8090)
base_url = f"http://{bind_ip}:8090/{run_id}"
```

**How it works:**
- Uses Python's `SimpleHTTPRequestHandler`
- Serves from `stages_out/` directory
- Agents download via HTTP: `http://IP:8090/run_id/stage_N.safetensors`

#### Step 7: Stage Loading
```python
# Instructs each agent to download and load its stage
for agent, stage_file in zip(agents, stage_files):
    call_stage_load(
        agent_url=f"http://agent:8008",
        stage_url=f"http://coordinator:8090/{run_id}/{stage_file}",
        stage_id="pc-stage0"
    )
```

**What happens on agent:**
1. Downloads stage file via HTTP (streaming)
2. Saves to cache: `~/.cache/ebp_stages/stage_id.safetensors`
3. Reads metadata (keys, shapes, dtypes) - **no tensor loading**
4. Returns metadata to coordinator

---

### 3. **DP Planner** (`ebp/planner_dp.py`)

**Algorithm: Dynamic Programming with Memory Constraints**

**Objective:**
Minimize bottleneck stage time subject to:
- **Hard constraint**: `stage_memory ≤ device_budget` (MUST be satisfied)
- **Soft objective**: Minimize `max(stage_time_i)` where `stage_time = compute_cost / GFLOPS`

**How it works:**

1. **Prefix Sums** (O(1) range queries):
   ```python
   ps_cost[i] = sum(layer_costs[0..i])
   ps_mem[i] = sum(layer_bytes[0..i])
   ```

2. **Feasibility Check** (for target bottleneck T):
   ```python
   def feasible(T):
       for each device:
           # Find max layers that fit in memory AND time ≤ T
           while memory_ok and time_ok:
               extend stage
       return all_layers_assigned
   ```

3. **Binary Search**:
   ```python
   # Find minimum T where feasible(T) == True
   lo = total_cost / sum_gflops
   hi = total_cost / min_gflops
   while lo < hi:
       mid = (lo + hi) / 2
       if feasible(mid):
           hi = mid  # Can do better
       else:
           lo = mid  # Need more time
   ```

4. **Reconstruction**:
   - Greedily assign layers at target T
   - Respects memory constraints at each step

**Key Features:**
- ✅ **Memory-constrained**: Hard limits prevent OOM
- ✅ **Optimal**: Finds best partition for given constraints
- ✅ **Fast**: O(L × D × log(T)) complexity

---

### 4. **Stage Packaging** (`ebp/package_stages.py`)

**Purpose:** Split model weights into per-device stage files

**Process:**

1. **Layer Pattern Detection**:
   ```python
   # Detects layer naming patterns
   patterns = [
       r"^model\.layers\.(\d+)\.",      # LLaMA/Qwen
       r"^transformer\.h\.(\d+)\.",     # GPT-2
   ]
   ```

2. **Tensor Assignment**:
   ```python
   for tensor_key in model:
       layer_id = extract_layer_id(key)  # e.g., "model.layers.5.attn.q.weight" → 5
       if layer_id in stage_range:
           assign_to_stage(tensor)
   ```

3. **Streaming Packaging**:
   ```python
   for stage in stages:
       stage_tensors = {}
       # Load in small batches
       for batch in batches:
           load_tensors(batch)
           stage_tensors.update(batch)
           gc.collect()  # Free memory
       save_file(stage_tensors, f"stage_{i}.safetensors")
   ```

**Memory Safety:**
- Batches of 5-20 tensors (not all at once)
- Garbage collection after each batch
- One stage at a time (don't accumulate multiple stages)

---

## 🔄 Complete Workflow Example

### Setup Phase

**1. Start Agents:**
```bash
# Terminal 1: PC Agent
python -m ebp.agent_main --name pc --port 8008

# Terminal 2: Pi Agent (via SSH)
ssh pi@172.20.10.2
python -m ebp.agent_main --name pi --port 8008
```

**2. Run Coordinator:**
```bash
python -m ebp.coordinator_main \
  --model-path /path/to/Qwen2.5-1.5B-Instruct \
  --urls "http://127.0.0.1:8008,http://172.20.10.2:8008" \
  --pipeline-order "pc,pi" \
  --mem-fraction 0.45 \
  --package
```

### Execution Flow

```
1. Coordinator → PC Agent: GET /v1/capabilities
   Response: {cpu: 2, ram_avail: 3.75GB, ...}

2. Coordinator → PC Agent: POST /v1/profile/matmul
   Response: {eff_gflops: 35.80}

3. Coordinator → Pi Agent: GET /v1/capabilities
   Response: {cpu: 4, ram_avail: 7.05GB, ...}

4. Coordinator → Pi Agent: POST /v1/profile/matmul
   Response: {eff_gflops: 30.71}

5. Coordinator: Analyzes model (reads config.json, safetensors metadata)
   Result: L=28, H=1536, total_bytes=2.88GB

6. Coordinator: Runs DP partitioner
   Result: PC gets layers [0..14], Pi gets [15..27]

7. Coordinator: Packages stages
   - Creates stage_0.safetensors (PC's layers)
   - Creates stage_1.safetensors (Pi's layers)

8. Coordinator: Starts HTTP server on port 8090
   Serving: stages_out/Qwen2.5-1.5B-Instruct_1764404816/

9. Coordinator → PC Agent: POST /v1/stage/load
   Body: {stage_url: "http://172.20.10.4:8090/.../stage_0.safetensors"}
   PC downloads and caches stage file

10. Coordinator → Pi Agent: POST /v1/stage/load
    Body: {stage_url: "http://172.20.10.4:8090/.../stage_1.safetensors"}
    Pi downloads and caches stage file

11. Done! Both agents have their assigned model layers loaded.
```

---

## 🧠 Key Design Decisions

### 1. **Memory-First Approach**
- **Problem**: Previous version only balanced compute, causing OOM
- **Solution**: Hard memory constraints in DP algorithm
- **Result**: Guarantees no device exceeds its budget

### 2. **Streaming Packaging**
- **Problem**: Loading entire model (2.88GB) causes OOM
- **Solution**: Process one stage at a time, small batches
- **Result**: Peak memory usage reduced by ~70%

### 3. **Metadata-Only Loading**
- **Problem**: Loading full tensors into RAM on low-memory devices
- **Solution**: Agents read safetensors metadata only
- **Result**: Can verify stage loaded without OOM risk

### 4. **Pipeline Parallelism**
- **Why**: Model parallelism (tensor parallelism) requires high bandwidth
- **Why**: Data parallelism requires model replication (too much memory)
- **Why Pipeline**: Natural fit for sequential transformer layers

---

## 📊 Memory Model

### Per-Device Memory Usage

```
Total RAM = System + Applications + Model Weights + KV Cache + Buffers
```

**Budget Calculation:**
```python
usable_ram = available_ram × mem_fraction  # Default: 45-60%
```

**Stage Memory:**
```python
stage_memory = sum(layer_bytes[layer_range])
```

**KV Cache (not yet implemented):**
```python
kv_cache = num_layers × 2 × num_heads × head_dim × bytes_per_elem × context_length
```

**Current Status:**
- ✅ Model weights: Accounted for
- ✅ Memory budgets: Enforced
- ⚠️ KV cache: Not yet included (future work)
- ⚠️ Activation memory: Estimated roughly

---

## 🔐 Safety Features

### 1. **Memory Constraints**
- Hard limits prevent OOM
- Validates before partitioning
- Clear error messages if model too large

### 2. **Streaming Processing**
- Batches prevent memory spikes
- Garbage collection between batches
- One stage at a time

### 3. **Error Handling**
- Network timeouts
- Retry logic (can be added)
- Graceful degradation

### 4. **Validation**
- Checks model path exists
- Validates agent reachability
- Verifies memory budgets are sufficient

---

## 🚀 Future Extensions

### Planned Features

1. **KV Cache Correctness**
   - Track KV cache memory per device
   - Ensure cache fits in budget
   - Handle context window limits

2. **Elastic Scaling**
   - "Borrow" cloud/MEC resources for stragglers
   - Dynamic re-partitioning
   - Load balancing

3. **Real Inference**
   - Currently only loads stages
   - Next: Implement forward pass across devices
   - Handle activations between stages

4. **Better Profiling**
   - Replace matmul with transformer-shaped benchmarks
   - Measure actual layer compute times
   - Network bandwidth measurement

---

## 📁 File Structure

```
LLM_project/
├── ebp/                    # Main package
│   ├── agent_main.py       # Agent entry point
│   ├── agent_app.py        # Agent FastAPI app
│   ├── coordinator_main.py # Coordinator entry point
│   ├── planner_dp.py       # DP partitioner
│   ├── package_stages.py   # Stage packaging
│   ├── serve.py            # HTTP file server
│   ├── models.py           # Pydantic data models
│   ├── common.py            # Utilities
│   └── discovery.py        # mDNS (optional)
├── stages_out/             # Generated stage files
├── plan.json               # Partition plan
├── requirements-pc.txt      # PC dependencies
├── requirements-agent.txt  # Agent dependencies
└── *.sh                    # Helper scripts
```

---

## 🎓 Key Concepts

### Pipeline Parallelism
- **Definition**: Split model layers across devices sequentially
- **Flow**: Input → Device 1 (layers 0-14) → Device 2 (layers 15-27) → Output
- **Communication**: Activations passed between devices
- **Advantage**: Works with low bandwidth (only activations, not weights)

### Memory-Constrained Partitioning
- **Goal**: Assign layers such that each device's stage fits in memory
- **Constraint**: `sum(layer_bytes[range]) ≤ device_budget`
- **Objective**: Minimize bottleneck (slowest stage)
- **Method**: Dynamic programming with binary search

### Safetensors Format
- **What**: Efficient tensor storage format
- **Features**: Fast loading, metadata access, safe (no arbitrary code)
- **Usage**: Read metadata without loading tensors (memory-safe)

---

## 🔧 Configuration

### Memory Fraction (`--mem-fraction`)
- **Default**: 0.45-0.50
- **Meaning**: Fraction of available RAM to use for model weights
- **Reserves**: Space for KV cache, OS, buffers
- **Adjust**: Lower = safer, Higher = more layers per device

### Min Prefix (`--min-prefix`)
- **Default**: 4
- **Meaning**: Minimum layers on first device
- **Why**: First device handles embeddings, often needs more layers
- **Use case**: Ensure embeddings stay on faster device

---

## 📈 Performance Characteristics

### Partitioning Time
- **Complexity**: O(L × D × log(T)) where L=layers, D=devices, T=time range
- **Typical**: < 1 second for 28 layers, 2 devices

### Packaging Time
- **Depends on**: Model size, number of stages, disk speed
- **Typical**: 1-5 minutes for 2.88GB model, 2 stages
- **Bottleneck**: Disk I/O (reading/writing safetensors)

### Network Transfer
- **Stage sizes**: ~1-2GB per stage
- **Bandwidth**: Depends on Wi-Fi (typically 20-100 Mbps)
- **Time**: 2-10 minutes per stage download

---

## 🐛 Common Issues & Solutions

### 1. **OOM During Packaging**
- **Cause**: Loading entire model into RAM
- **Fix**: Use smaller batches, lower mem_fraction, or skip packaging for single device

### 2. **404 File Not Found**
- **Cause**: File server URL mismatch (now fixed)
- **Fix**: Serve from parent directory, include run_id in URL path

### 3. **Model Too Large**
- **Cause**: Model exceeds device memory budgets
- **Fix**: Use more devices, reduce mem_fraction, or use smaller model

### 4. **Agent Not Reachable**
- **Cause**: Network issues, firewall, agent not running
- **Fix**: Check connectivity, start agent, verify firewall rules

---

## 🎯 Summary

**What This Project Does:**
Takes a large LLM model that doesn't fit on a single device and intelligently splits it across multiple devices using pipeline parallelism, ensuring each device has enough memory and the workload is balanced.

**Key Innovation:**
Memory-constrained dynamic programming partitioner that guarantees no OOM while optimizing for compute balance.

**Current Status:**
✅ Partitioning works
✅ Packaging works (with memory safety)
✅ Stage loading works
⏳ Inference (forward pass) - not yet implemented

**Next Steps:**
1. Implement actual inference across devices
2. Add KV cache memory tracking
3. Add elastic scaling capabilities

