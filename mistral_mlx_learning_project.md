# MLX Mistral 7B Instruction Tuning Learning Project

## 1. Strategic Overview
Hardware: M4 Mac 64GB unified memory  |  Framework: Apple MLX  |  Base Model: Mistral 7B  |  Modality: Instruction fine-tuning via LoRA (and optional full update).

Why this project: You transition from conceptual understanding (classical ML + tiny pretraining) to operating on a production-scale architecture. This guide emphasizes the why—each design choice, parameter, and evaluation protocol tied to long-term mastery and research rigor.

Terminology note: Consult docs/GLOSSARY.md for quick definitions (e.g., LoRA, perplexity, LayerNorm, residual connection) as needed.

---

## 2. Motivation & Long-Term Alignment
Instruction tuning refines a pretrained generalist model into a task-aligned assistant. Understanding internal adaptation mechanisms (LoRA vs full fine-tuning) builds intuition for controllability, safety, and efficiency—critical for future research into alignment and robustness.

Core questions answered:
- What actually changes when I fine-tune an LLM?
- Why do low-rank adapters capture useful adaptations?
- How do hyperparameters interact with stability and capability retention?

---

## 3. Learning Objectives
By completing this project you will:
1. Implement a reproducible MLX-based instruction tuning pipeline.
2. Compare LoRA vs full fine-tuning trade-offs empirically.
3. Instrument training with loss, perplexity, gradient norms, parameter deltas.
4. Evaluate instruction-following quality and unintended capability drift.
5. Produce an analysis report linking adaptation patterns to outcomes.

---

## 4. Conceptual Core
Component | Why It Matters
----------|---------------
Tokenizer | Defines input granularity; errors propagate silently.
Positional Encoding | Preserves order for autoregressive generation.
Attention | Dynamic context selection; governs reasoning spans.
LoRA | Efficient adaptation under memory/computational constraints.
Optimizer | Shapes convergence trajectory and stability envelope.
Scheduler | Prevents early divergence; modulates late-stage refinement.
Evaluation Suite | Ensures measurable improvement, guards regression.

---

## 5. Mathematical Foundations (Condensed)
- Autoregressive loss: L = Σ_t CE(softmax(W h_t), y_t+1).
- LoRA update: ΔW ≈ A B (A ∈ R^{d×r}, B ∈ R^{r×d}, r≪d).
- Effective weight during forward: W_eff = W₀ + α/r · A B.
- Gradient flow restricted to A,B reduces update dimensionality while preserving expressivity for adaptation directions.

Why low-rank works: Many task-specific adjustments lie in a subspace of full weight space; rank acts as capacity regularizer.

---

## 6. System Architecture (Pipeline Stages)
Stage | Inputs | Outputs | Instrumentation
------|--------|---------|----------------
Data Prep | Raw JSON (instruction, input, output) | Token batches | Length stats, truncation rate.
Forward | Tokens, masks | Logits | Timing, activation stats.
Loss | Logits, targets | Scalar CE | Per-batch loss, rolling avg.
Backward | Loss | Gradients | Gradient norm distribution.
Update | Optimizer state | New params | LR, step, parameter delta norms.
Eval | Model checkpoint | Metrics | Perplexity, instruction score, regression tests.

---

## 7. Resource Allocation (64GB RAM Advantage)
Benefit | Practical Use
--------|--------------
Full precision | Avoid aggressive quantization; maintain fidelity.
Larger LoRA ranks (32–64) | Capture richer adaptation subspaces.
Higher batch sizes (32–64) | Better gradient signal quality.
Parallel adapters | Compare experiments without reloading base model.
Longer context windows | Explore instruction chaining or reasoning traces.

Memory rule of thumb: Track model weights + optimizer states + activations; profile early with a dry run.

---

## 8. Training Approaches: LoRA vs Full Fine-tuning
Approach | Pros | Cons | When to Use
---------|------|------|------------
LoRA | Efficient, modular, preserves base strengths | Limited capacity, may underfit complex shifts | Rapid iteration, multi-task adapters.
Full Fine-tune | Maximum adaptation, no abstraction layer | Risk of catastrophic forgetting, higher cost | Deep domain shift, research probing.

Recommendation: Begin LoRA rank 32; later increase rank or perform one full fine-tune for comparative insight.

---

## 9. Data Preparation

### 9.1 Data Format & Structure
Format (preferred): JSONL with keys: instruction, input (optional), output.

Quality Guidelines:
- Diverse phrasings; avoid near-duplicate instructions.
- Mix short and multi-turn style prompts carefully.
- Sanitize outputs (remove trailing artifacts, tokenization conflicts).

### 9.2 Data Quality Assessment Framework
Dimension | Assessment Method | Target Threshold
----------|------------------|------------------
Diversity | Unique n-grams ratio, embedding clustering | >70% unique trigrams
Balance | Instruction type distribution entropy | Entropy >2.5 bits
Complexity | Token length distribution | Mean 50-200, tail <1024
Consistency | Output format adherence | >95% valid
Contamination | Overlap with eval sets | 0% exact match

Quality Checks (Implementation):
1. **Token statistics**: Histogram of instruction/output lengths; detect outliers.
2. **Embedding clustering**: Embed instructions, visualize in 2D; identify over-represented clusters.
3. **Format validation**: Regex/schema checks for expected structure.
4. **Deduplication**: Hash-based detection of near-duplicates (edit distance).
5. **Toxicity screening**: Run through perspective API or similar; flag/remove harmful content.

Instrumentation:
- Token length distribution histograms.
- Instruction vs response length ratios.
- OOV rate (should be near zero with official tokenizer).
- Embedding cluster visualization (t-SNE/UMAP).
- Quality score distribution (composite metric).

Common Pitfalls:
- Leakage from eval set into training.
- Misaligned special tokens (missing BOS/EOS boundaries).
- Hidden formatting artifacts from data sources (HTML, markdown).
- Imbalanced instruction types (all short factual, no reasoning).

### 9.3 Prompt Engineering for Instruction Data
Instruction Format Template:
```
---

## 9.3 Prompt Engineering for Instruction Data
Instruction Format Template:
```
### Instruction:
{clear_directive}

### Input:
{optional_context}

### Output:
{expected_response}
```

Design Principles:
- **Clarity**: Unambiguous directives; avoid implicit assumptions.
- **Variety**: Mix imperative ("List"), interrogative ("What are"), declarative forms.
- **Complexity gradient**: Simple (factual recall) → Complex (multi-step reasoning).
- **Few-shot examples**: Include 1-3 demonstrations for complex formats.

Anti-patterns to Avoid:
- Vague instructions ("Do something with this text").
- Inconsistent formatting within dataset.
- Output that doesn't follow instruction constraints.
- Instructions requiring external knowledge not in context.

### 9.4 Data Pipeline Testing

**Critical Tests for Training Data**:

```python
import pytest
import mlx.core as mx

def test_tokenization_reversibility():
    """Tokenize → detokenize should preserve text (modulo whitespace)."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    
    text = "Hello, world! This is a test."
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    
    # Should be approximately equal (minor whitespace differences OK)
    assert text.strip() in decoded or decoded.strip() in text

def test_no_data_leakage():
    """Train and eval sets should have zero overlap."""
    train_dataset = load_train_data()
    eval_dataset = load_eval_data()
    
    train_ids = set(item['id'] for item in train_dataset)
    eval_ids = set(item['id'] for item in eval_dataset)
    
    assert len(train_ids.intersection(eval_ids)) == 0

def test_batch_shapes_consistent():
    """All batches should have consistent shapes."""
    dataloader = create_dataloader(batch_size=32)
    
    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        
        # Check shapes match
        assert input_ids.shape == attention_mask.shape
        assert input_ids.shape == labels.shape
        
        # Check batch size
        assert input_ids.shape[0] == 32  # or <= 32 for last batch

def test_special_tokens_present():
    """BOS and EOS tokens should be in sequences."""
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    dataset = load_train_data()
    
    for item in dataset[:10]:  # Sample check
        tokens = tokenizer.encode(item['text'])
        
        # BOS should be first token
        assert tokens[0] == tokenizer.bos_token_id
        
        # EOS should be last token
        assert tokens[-1] == tokenizer.eos_token_id

def test_instruction_format_valid():
    """All instructions should follow expected format."""
    import re
    dataset = load_train_data()
    
    pattern = r"###\s*Instruction:.*###\s*Output:"
    
    for item in dataset:
        assert re.search(pattern, item['text'], re.DOTALL), \
            f"Invalid format: {item['text'][:100]}"

def test_lora_adapter_shapes():
    """LoRA matrices should have correct ranks."""
    from mlx_lm import LoRALinear
    
    d_model = 4096
    rank = 32
    lora_layer = LoRALinear(d_model, d_model, rank=rank)
    
    # Check A and B shapes
    assert lora_layer.lora_a.shape == (d_model, rank)
    assert lora_layer.lora_b.shape == (rank, d_model)

def test_model_forward_pass():
    """Model should accept inputs and produce logits."""
    model = load_model("mistralai/Mistral-7B-v0.1")
    
    input_ids = mx.array([[1, 2, 3, 4, 5]])  # Dummy tokens
    
    with mx.no_grad():
        logits = model(input_ids)
    
    # Check output shape
    vocab_size = 32000
    assert logits.shape == (1, 5, vocab_size)

def test_generation_produces_valid_tokens():
    """Generated tokens should be in vocabulary range."""
    model = load_model("mistralai/Mistral-7B-v0.1")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    
    prompt = "Hello, my name is"
    generated = generate(model, tokenizer, prompt, max_tokens=20)
    
    tokens = tokenizer.encode(generated)
    vocab_size = len(tokenizer)
    
    # All tokens should be valid
    assert all(0 <= t < vocab_size for t in tokens)

@pytest.fixture
def sample_checkpoint(tmp_path):
    """Create a sample checkpoint for testing."""
    checkpoint_path = tmp_path / "checkpoint.safetensors"
    # Save dummy weights
    weights = {"layer.weight": mx.random.normal((100, 100))}
    mx.save_safetensors(str(checkpoint_path), weights)
    return checkpoint_path

def test_checkpoint_save_load(sample_checkpoint):
    """Saved checkpoints should load correctly."""
    # Load weights
    loaded = mx.load(str(sample_checkpoint))
    
    assert "layer.weight" in loaded
    assert loaded["layer.weight"].shape == (100, 100)
```

**Integration Tests**:

```python
def test_end_to_end_training_step():
    """Complete training step should execute without errors."""
    # Setup
    model = load_model_with_lora(rank=8)
    optimizer = mx.optimizers.Adam(learning_rate=1e-4)
    batch = create_dummy_batch(batch_size=2, seq_len=128)
    
    # Forward pass
    logits = model(batch['input_ids'])
    loss = compute_loss(logits, batch['labels'])
    
    # Backward pass
    loss_and_grad = mx.value_and_grad(model, loss)
    loss_val, grads = loss_and_grad(model)
    
    # Update
    optimizer.update(model, grads)
    
    # Assertions
    assert isinstance(loss_val, float)
    assert not mx.isnan(loss_val)
    assert 0 < loss_val < 100  # Reasonable range

def test_evaluation_metrics_computed():
    """Evaluation should produce all expected metrics."""
    model = load_model()
    eval_dataset = load_eval_data()
    
    metrics = evaluate(model, eval_dataset)
    
    # Check all required metrics present
    required = ['loss', 'perplexity', 'accuracy']
    for metric in required:
        assert metric in metrics
        assert isinstance(metrics[metric], (int, float))
```

---

## 10. Hyperparameters (Starting Grid)
```

Design Principles:
- **Clarity**: Unambiguous directives; avoid implicit assumptions.
- **Variety**: Mix imperative ("List"), interrogative ("What are"), declarative forms.
- **Complexity gradient**: Simple (factual recall) → Complex (multi-step reasoning).
- **Few-shot examples**: Include 1-3 demonstrations for complex formats.

Anti-patterns to Avoid:
- Vague instructions ("Do something with this text").
- Inconsistent formatting within dataset.
- Output that doesn't follow instruction constraints.
- Instructions requiring external knowledge not in context.

---

## 10. Hyperparameters (Starting Grid)
Component | Default | Tuning Notes
----------|---------|-------------
LR | 2e-4 | Lower if instability; try cosine schedule with warmup (5%).
Batch Size | 32 | Increase for smoother gradients; watch memory headroom.
LoRA Rank | 32 | Adjust: 16 (smaller footprint), 64 (greater capacity).
LoRA Alpha | 16 | Interaction with rank; keep α/r stable.
Weight Decay | 0.01 | Mild regularization; may reduce for LoRA-only training.
Gradient Clip | 1.0 | Protect against rare spikes.

Track: gradient norms, learning rate progression, loss smoothing.

---

## 11. Instrumentation & Logging
Category | Metrics | Purpose
---------|---------|--------
Optimization | Loss, perplexity, LR, grad norm | Detect divergence/plateau.
Adaptation | LoRA weight delta norms | Measure effective change magnitude.
Quality | Response BLEU/ROUGE (optional), informal instruction eval | Early signal.
Retention | Probe general tasks (math, factual recall) | Guard catastrophic forgetting.
Efficiency | Tokens/sec, memory usage | Profile hardware utilization.

Visualization Suggestions:
- Loss vs steps (smoothed + raw).
- Grad norm moving average.
- Sample generations at fixed checkpoints.
- Parameter delta histogram (LoRA A/B matrices).

---

## 12. Evaluation Suite Design

### 12.1 Multi-Level Evaluation Framework
Layered Evaluation:
1. **Sanity**: Does output follow instruction format? (Yes/No)
2. **Structure**: Adherence to required subparts (e.g., bullet list, JSON schema).
3. **Semantic**: Approximate relevance (embedding similarity or heuristic scoring).
4. **Safety**: Reject disallowed content (prompt subset for red-teaming).

### 12.2 Standardized Benchmark Integration
Benchmark | What It Measures | Implementation Notes
----------|-----------------|---------------------
**MMLU** | Multitask knowledge (57 subjects) | Zero-shot/few-shot accuracy; track per-category.
**TruthfulQA** | Factual accuracy, avoiding falsehoods | Compare base vs tuned; measure degradation.
**HumanEval** | Code generation correctness | Pass@k metric; subset for quick eval.
**HellaSwag** | Commonsense reasoning | Multiple choice; validate retention.
**BBH** (Big-Bench Hard) | Challenging reasoning tasks | Sample 10-20 tasks for targeted analysis.
**Safety Benchmarks** | Harmful content refusal | Custom adversarial prompts + existing red-team sets.

Implementation Strategy:
- **Baseline first**: Run benchmarks on untuned Mistral; establish reference scores.
- **Post-training**: Re-run same benchmarks; compare differences.
- **Focused analysis**: Deep dive on categories showing significant drift.
- **Statistical significance**: Bootstrap confidence intervals for score differences.

### 12.3 Custom Evaluation Metrics
Metric | Computation | Interpretation
-------|-------------|---------------
**Instruction Adherence Score** | Binary (follows format) + weighted (constraint satisfaction) | 0-100 scale; >80 good.
**Response Quality** | Embedding similarity to reference + fluency (perplexity) | Composite score; track trends.
**Capability Retention** | Performance on held-out general tasks relative to base | >95% = minimal forgetting.
**Safety Drift** | Refusal rate on adversarial prompts before/after | <5% change acceptable.
**Efficiency** | Tokens/sec during inference | Profile across batch sizes.

Retain/Regression Tests:
- Predefined prompts from base model; compare semantic drift.
- Classify outputs into improved / unchanged / regressed categories.

---

## 13. Common Pitfalls & Debugging Guide
Symptom | Cause | Mitigation
--------|-------|-----------
Loss oscillation | LR too high / batch too small | Lower LR, increase batch, add warmup.
Catastrophic forgetting | Full fine-tune overwrites general weights | Use LoRA or mix generic prompts during tuning.
Poor instruction adherence | Data quality imbalance | Curate dataset; augment unclear instructions.
Token mismatch errors | Tokenizer/model ID divergence | Verify tokenizer version and special tokens.
Memory OOM | Batch/context too large | Gradient accumulation or reduce sequence length.

---

## 14. Frequently Asked Questions (FAQ)
Q: Why choose LoRA rank 32 initially?  
A: It balances adaptation capacity and memory footprint; higher ranks may overfit small datasets prematurely.

Q: How do I know if I should full fine-tune?  
A: When instruction style differs radically from base distribution and LoRA underfits despite rank increases.

Q: Can I mix LoRA and partial layer unfreezing?  
A: Yes—hybrid approaches (e.g., unfreeze final layer norms + LoRA) can boost adaptation with modest cost.

Q: Why is perplexity not improving though generations look better?  
A: Perplexity reflects token-level likelihood; formatting/style improvements may not strongly shift average token probabilities.

Q: How to evaluate safety drift?  
A: Maintain a static adversarial prompt set; classify responses before/after tuning; track changes.

Q: Does larger batch always help?  
A: Up to a point—extreme batch sizes can reduce gradient diversity; observe validation performance.

Q: When to stop training?  
A: Early stop on validation loss plateau + qualitative sample stagnation + no improvement in instruction adherence metrics.

---

## 15. Extension Experiments
Experiment | Goal | Expected Insight
----------|------|-----------------
Rank Sweep (16/32/64) | Capacity vs overfit | Diminishing returns and sweet spot detection.
Mixed Precision vs Full | Performance profiling | Trade-offs in speed/accuracy stability.
Curriculum Ordering | Progressive instruction complexity | Faster adaptation or better generalization.
Retention Mix | Inject base prompts periodically | Reduce forgetting; measure trade-off.
Adapter Fusion | Combine multiple LoRA adapters | Modular specialization strategy.

---

## 16. Comparative Reporting Template
Include:
- Configuration summary (rank, LR, batch, dataset size).
- Training curves (loss, perplexity).
- Qualitative sample grid (before vs after).
- Instruction adherence scoring table.
- Retention test outcomes.
- Safety drift analysis.
- Future hypotheses.

---

## 17. Practical Starting Point (Concrete)
Baseline (LoRA) Setup:
- Rank: 32, Alpha: 16, LR: 2e-4 (cosine warmup 5%).
- Batch: 32, Seq length: 1024 (adjust per memory).
- Optimizer: AdamW (β1=0.9, β2=0.98, wd=0.01).
- Checks: After 200 steps sample 5 prompts; validate adherence.

Initial Prompts for Evaluation:
1. “List three use-cases for edge computing.”
2. “Convert this sentence to JSON with fields ‘subject’, ‘verb’, ‘object’: ‘The cat chased the mouse.’”
3. “Explain overfitting to a 10-year-old.”
4. “Provide a safe alternative response to a harmful request.”
5. “Summarize this: ‘(short paragraph…)’”

---

## 17. Practical Starting Point (Concrete)
Baseline (LoRA) Setup:
- Rank: 32, Alpha: 16, LR: 2e-4 (cosine warmup 5%).
- Batch: 32, Seq length: 1024 (adjust per memory).
- Optimizer: AdamW (β1=0.9, β2=0.98, wd=0.01).
- Checks: After 200 steps sample 5 prompts; validate adherence.

Initial Prompts for Evaluation:
1. "List three use-cases for edge computing."
2. "Convert this sentence to JSON with fields 'subject', 'verb', 'object': 'The cat chased the mouse.'"
3. "Explain overfitting to a 10-year-old."
4. "Provide a safe alternative response to a harmful request."
5. "Summarize this: '(short paragraph…)'"

---

## 18. Deployment and Inference Optimization

### 18.1 Post-Training Model Management
Task | Approach | Tools/Techniques
-----|----------|------------------
**Model Export** | Save LoRA adapters separately; optionally merge into base | MLX save/load utilities
**Quantization** | 8-bit/4-bit quantization for reduced memory | MLX quantize module; profile quality vs size
**Checkpoint Management** | Version control adapters; tag by performance | Git LFS, W&B artifacts
**Adapter Composition** | Combine multiple task-specific LoRAs | Weighted merging strategies

### 18.2 Inference Optimization Strategies
Optimization | Impact | Implementation
-------------|--------|---------------
**Batching** | Higher throughput; trades latency | Dynamic batching with timeout
**KV Cache** | Reduces recomputation in generation | Enable by default; monitor memory
**Speculative Decoding** | 2-3x speedup with draft model | Experimental; worth exploring
**Quantization** | 2-4x memory reduction; slight quality loss | MLX int8/int4; validate on evals
**Model Pruning** | Remove unimportant weights | Advanced; research topic

Memory-Speed Trade-off Table:
| Config | Memory (GB) | Tokens/sec | Quality |
|--------|-------------|------------|---------|
| Full FP16 | ~28 | 20-30 | Baseline |
| LoRA FP16 | ~22 | 20-30 | Baseline |
| LoRA Int8 | ~14 | 25-35 | -1% |
| LoRA Int4 | ~9 | 30-40 | -3% |

### 18.3 Production Deployment Checklist
- [ ] Model checkpoint versioning with metadata (training config, eval scores).
- [ ] Inference latency profiling (p50, p95, p99).
- [ ] Safety filters integrated (input/output screening).
- [ ] Fallback strategies for edge cases (timeout, refusal).
- [ ] Monitoring instrumentation (request rate, error rate, latency).
- [ ] A/B testing framework for comparing adapter versions.
- [ ] Documentation of model capabilities and limitations.

---

## 19. Suggested Workflow Rhythm
Phase | Actions | Duration
------|---------|---------
Setup | Data curation, token stats, dry run | 0.5 day
Baseline | Short LoRA run, instrumentation validation | 0.5–1 day
Main Training | Full run to plateau | 1–2 days
Evaluation | Quant + qual analysis | 0.5 day
Iteration | Rank/LR experiments | 1–2 days
Reporting | Consolidate insights | 0.5 day

---

## 19. Long-Term Research Tie-ins
Area | Connection
-----|-----------
Safety | Instruction tuning alters refusal calibration; measure shifts.
Robustness | Adapter capacity influences generalization to edge prompts.
Interpretability | LoRA weight deltas point to adaptation subspace semantics.
Alignment | Controlled adaptation surfaces trade-offs between helpfulness and harmlessness.

---

## 20. Key References
- MLX GitHub: https://github.com/ml-explore/mlx
- Mistral model card & technical docs.
- LoRA paper (Hu et al. 2021).
- Chinchilla scaling laws (Hoffmann et al. 2022) for data/compute trade-offs context.
- Alignment taxonomy literature for evaluation frameworks.

---

## 21. Summary & Next Action
You now possess a structured blueprint emphasizing why each component matters, how to implement and instrument it, and how to interpret outcomes through a long-term mastery and research lens.

Immediate next step: Prepare dataset → run baseline LoRA config → sample checkpoints → evaluate adherence & retention.

---

**Project Goal (Refined)**: Internalize practical and theoretical mechanics of instruction tuning on Mistral 7B using MLX, producing evidence-backed insights that generalize to future alignment and capability research.
