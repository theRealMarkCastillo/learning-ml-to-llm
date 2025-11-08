# Complete ML Learning Foundation
## From Classical ML Through Pretraining to LLM Fine-tuning

### Overview
This expanded learning path builds from classical ML fundamentals through understanding pretraining from scratch, culminating in instruction tuning Mistral 7B. The middle pretraining section is crucial—it shows you what a "base model" actually is and why fine-tuning works.

**Learning Arc**: Classical ML fundamentals → understand optimization deeply → build a tiny transformer → pretrain it from scratch → understand what you learned → fine-tune Mistral with full clarity

Terminology helper: New to LLM specifics (attention heads, positional encodings, perplexity, LoRA, tokenization)? Refer to docs/GLOSSARY.md throughout this section.

**Hardware Note**: Your M4 with 64GB RAM can comfortably handle all of this. Pretraining on small corpus takes hours to a day, which is very reasonable.

---

## PART 1: CLASSICAL ML FOUNDATIONS
### (Projects 1-11, unchanged from previous document)

Follow the classical ML learning path as outlined previously. This builds your optimization and experimentation foundations.

Key projects:
- Projects 1-4: Fundamentals from scratch (linear regression, logistic regression, multi-class, regularization)
- Projects 5-7: Standard algorithms (trees, forests, SVMs, metrics)
- Projects 8-11: Validation strategies and end-to-end pipeline

**Timeline**: 8-12 weeks

**Outcome**: Deep understanding of gradient descent, loss functions, generalization, evaluation. You'll have trained dozens of models and understand why they work.

---

## PART 2: UNDERSTANDING TRANSFORMERS AND PRETRAINING
### The Critical Middle Section

This is where you build a small transformer from scratch and pretrain it. This section transforms your understanding from “fine-tuning works somehow” to “I see exactly what happens.” Each project includes motivation, objectives, math, steps, instrumentation, pitfalls, FAQs, and long-term alignment.

---

## Project 12: Transformer Architecture from Scratch

#### Motivation (Why)
Demystify the core building blocks underpinning modern LLMs: [attention](docs/GLOSSARY.md#query--key--value) (with [heads](docs/GLOSSARY.md#head-attention-head)), [residual connections](docs/GLOSSARY.md#residual-connection), [LayerNorm](docs/GLOSSARY.md#layernorm), and [feed-forward](docs/GLOSSARY.md#feed-forward-network-ffn) layers.

#### Learning Objectives
- Implement multi-head self-attention and feed-forward blocks.
- Assemble a decoder-only transformer and verify tensor shapes.
- Understand residual pathways and normalization’s role in stability.

#### Conceptual Core
- [Attention](docs/GLOSSARY.md#query--key--value) as content-based routing of information across positions.
- Multi-[heads](docs/GLOSSARY.md#head-attention-head) capture diverse relational patterns; [residuals](docs/GLOSSARY.md#residual-connection) stabilize depth.
- [Positional encodings](docs/GLOSSARY.md#positional-encoding) restore order awareness.

#### Mathematical Foundations
- Attention(Q, K, V) = [softmax](docs/GLOSSARY.md#softmax)(QKᵀ/√d_k) V.
- [LayerNorm](docs/GLOSSARY.md#layernorm)(x) = (x−μ)/σ · γ + β (per feature normalization).
- [Residual](docs/GLOSSARY.md#residual-connection) block: x + f(x) improves optimization.

#### Implementation Steps (How)
1. Implement scaled dot-product attention with [masking](docs/GLOSSARY.md#mask-attention-mask).
2. Build multi-head wrapper; concat heads and project.
3. Add MLP (FFN) with [GELU](docs/GLOSSARY.md#gelu)/SiLU; residual + [LayerNorm](docs/GLOSSARY.md#layernorm).
4. Stack blocks; add token and [positional embeddings](docs/GLOSSARY.md#positional-encoding).
5. Run a dummy forward pass; check shapes and logits.

#### Instrumentation & Evaluation
- Assert tensor shapes at each step; unit tests for attention.
- Gradient checks on small inputs.
- Parameter count vs configuration sanity checks.

**Comprehensive Testing Strategy**:

```python
import torch
import pytest

def test_attention_output_shape():
    """Attention should preserve sequence length."""
    batch, seq_len, d_model = 2, 10, 64
    attention = MultiHeadAttention(d_model, num_heads=4)
    x = torch.randn(batch, seq_len, d_model)
    output = attention(x)
    assert output.shape == (batch, seq_len, d_model)

def test_causal_mask_prevents_future_access():
    """Causal mask should zero out future positions."""
    seq_len = 5
    attention_weights = torch.rand(1, 1, seq_len, seq_len)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    
    attention_weights.masked_fill_(mask, float('-inf'))
    attention_probs = torch.softmax(attention_weights, dim=-1)
    
    # Future positions should be zero
    for i in range(seq_len):
        assert torch.allclose(
            attention_probs[0, 0, i, i+1:], 
            torch.zeros(seq_len - i - 1)
        )

def test_positional_encoding_uniqueness():
    """Each position should have unique encoding."""
    pos_encoding = PositionalEncoding(d_model=64, max_len=100)
    encodings = pos_encoding.pe[0]  # (max_len, d_model)
    
    # No two positions should be identical
    for i in range(10):
        for j in range(i+1, 10):
            assert not torch.allclose(encodings[i], encodings[j])

def test_transformer_overfit_tiny_batch():
    """Sanity check: model should overfit 2 samples."""
    model = TransformerDecoder(vocab_size=50, d_model=64, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Tiny batch
    x = torch.randint(0, 50, (2, 10))
    y = torch.randint(0, 50, (2, 10))
    
    # Train for 100 steps
    for _ in range(100):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 50), y.reshape(-1)
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Should achieve near-zero loss on 2 samples
    final_loss = loss.item()
    assert final_loss < 0.5, f"Failed to overfit: loss={final_loss}"

def test_gradient_flow():
    """All parameters should receive gradients."""
    model = TransformerDecoder(vocab_size=50, d_model=64, num_layers=2)
    x = torch.randint(0, 50, (2, 10))
    y = torch.randint(0, 50, (2, 10))
    
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 50), y.reshape(-1)
    )
    loss.backward()
    
    # Check all parameters have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
```

#### Common Pitfalls & Debugging
- Masking mistakes leak future tokens; verify with simple sequences.
- Scale mismatch (forgetting √d_k) harms training stability.
- Wrong broadcasting across batch/heads/sequence dims.

#### FAQ
- Q: Why decoder-only? A: Next-token prediction aligns with pretraining and simplifies scope.
- Q: Why LayerNorm over BatchNorm? A: Sequence modeling benefits from per-sample normalization; variable lengths complicate BatchNorm.
- Q: Do I need rotary/relative position? A: Not to learn—start with sinusoidal; explore rotary later.

#### Alignment to Long-Term Goals
You'll recognize these same components (with larger dims) inside Mistral; this removes black-box mystique during fine-tuning.

---

## Project 12.5: Embeddings Deep Dive

#### Motivation (Why)
Embeddings are the foundation of representation learning—understanding how continuous vectors capture semantic relationships is essential for interpreting and debugging LLMs.

#### Learning Objectives
- Implement and train word embeddings (Word2Vec skip-gram or simple factorization).
- Visualize embedding spaces and discover semantic clusters.
- Understand the geometry of meaning: analogies, distances, and directions.
- Connect token embeddings to positional embeddings and their joint role.

#### Conceptual Core
- Embeddings map discrete tokens to continuous vectors where proximity reflects semantic similarity.
- Training objective shapes the space: co-occurrence patterns create geometric structure.
- Vector arithmetic (king - man + woman ≈ queen) emerges from distributional regularities.
- Positional embeddings add sequence order information orthogonally to content.

#### Mathematical Foundations
- Word2Vec skip-gram: maximize log P(context|word) = softmax(u_context · v_word).
- Matrix factorization view: W ≈ UV^T where rows are embeddings.
- Cosine similarity: cos(θ) = (u·v)/(||u|| ||v||) measures semantic relatedness.
- Distance metrics: Euclidean vs cosine; cosine invariant to magnitude.

#### Implementation Steps (How)
1. Build a simple embedding layer: token IDs → vectors (random init).
2. Train on a small corpus (e.g., skip-gram on sentences).
3. Visualize embeddings with PCA/t-SNE in 2D; color by word type/frequency.
4. Test analogies: find vec(king) - vec(man) + vec(woman) nearest to vec(queen).
5. Compare learned vs random embeddings on downstream task (sentiment classification).

#### Instrumentation & Evaluation
- 2D scatter plots: cluster words by semantic category.
- Nearest neighbor tables for sample words.
- Analogy accuracy on standard test sets (capital-country, gender, etc.).
- Downstream task performance: embeddings as features.

#### Common Pitfalls & Debugging
- Insufficient training: embeddings collapse or lack structure.
- Vocabulary size vs embedding dim mismatch: too small dims limit expressivity.
- Ignoring frequency: rare words have noisy embeddings; consider subsampling.
- Evaluation bias: analogies reflect corpus biases (gender, culture).

#### FAQ
- Q: Why do embeddings capture meaning? A: Distributional hypothesis: words in similar contexts have similar meanings; training encodes co-occurrence patterns geometrically.
- Q: How many dimensions? A: 50-300 typical for words; transformers use 512-4096+; trade-off between capacity and computation.
- Q: Pretrained vs trained from scratch? A: Pretrained (Word2Vec, GloVe, transformer embeddings) capture broad semantics; task-specific training adapts.
- Q: How do positional embeddings work with token embeddings? A: Added element-wise; model learns to disentangle content from position.

#### Extensions & Next Experiments
- Compare Word2Vec, GloVe, and fastText embeddings.
- Contextualized embeddings (ELMo-style): same word, different vectors per context.
- Multilingual embeddings and cross-lingual transfer.
- Probing tasks: can you predict POS tags, syntax from embeddings?

#### Alignment to Long-Term Goals
LLM token embeddings are the input layer; understanding their role clarifies how models process language; fine-tuning can update embeddings (vocabulary expansion) or freeze them; visualization aids interpretability of model behavior.

---

## Project 13: Tokenization and Text Preprocessing

#### Motivation (Why)
Models operate on token IDs, not raw text; tokenization determines efficiency and vocabulary granularity.

#### Learning Objectives
- Compare character-, word-, and subword-level tokenization.
- Use a modern tokenizer (SentencePiece/BPE) and understand special tokens.
- Build a streaming dataloader for variable-length sequences.

#### Conceptual Core
- Subword [tokenization](docs/GLOSSARY.md#tokenizer) balances [vocabulary size](docs/GLOSSARY.md#vocabulary-vocab) and coverage.
- Sequence length, [padding](docs/GLOSSARY.md#padding), and [masking](docs/GLOSSARY.md#mask-attention-mask) impact compute and learning.

#### Mathematical Foundations
- [Byte Pair Encoding (BPE)](docs/GLOSSARY.md#byte-pair-encoding-bpe) merges frequent pairs to minimize corpus encoding length.
- [Perplexity](docs/GLOSSARY.md#perplexity) relates to [cross-entropy](docs/GLOSSARY.md#cross-entropy-loss) over token distributions.

#### Implementation Steps (How)
1. Train/load a tokenizer; inspect tokenization of sample text.
2. Build dataset → batches; handle padding and causal masks.
3. Expose BOS/EOS/PAD/UNK handling in pipeline.

#### Instrumentation & Evaluation
- Token length histograms; out-of-vocabulary analysis.
- Compare vocab sizes (4k/8k/16k) vs sequence lengths and loss.

#### Common Pitfalls & Debugging
- Mismatched tokenizer/model embeddings dims.
- Ignoring special tokens; misaligned labels by one position.
- Excessive padding reduces effective batch compute.

#### FAQ
- Q: Why BPE over word-level? A: Robust to rare words; smaller vocab; better generalization.
- Q: How big should vocab be? A: Start 5k–10k for tiny models; trade memory vs fragmentation.
- Q: Do I need BOS/EOS? A: For generation and sequence boundaries—yes.

#### Alignment to Long-Term Goals
Tokenization mismatches are a frequent source of fine-tuning bugs; this project inoculates against them.

---

## Project 13.5: Attention Visualization and Analysis

#### Motivation (Why)
Attention patterns reveal what the model focuses on during prediction—critical for debugging, interpretability, and understanding model reasoning.

#### Learning Objectives
- Extract and visualize attention weights from trained transformer.
- Identify patterns: local vs global attention, head specialization.
- Analyze attention entropy and its relationship to confidence.
- Diagnose attention pathologies (e.g., attending to delimiters).

#### Conceptual Core
- Attention weights show which positions influence each prediction.
- Different heads learn different patterns (syntax, coreference, positional).
- High entropy (diffuse attention) suggests uncertainty or reliance on multiple contexts.
- Low entropy (peaked attention) indicates focused dependence on specific tokens.

#### Mathematical Foundations
- Attention weights: α_ij = softmax_j(q_i · k_j / √d_k).
- Entropy: H = -Σ α_ij log α_ij measures attention spread.
- Head importance: ablate head, measure performance drop.

#### Implementation Steps (How)
1. Modify transformer from Project 12 to return attention weights.
2. Generate predictions on sample sentences; extract all layer/head attentions.
3. Plot heatmaps: rows=queries, cols=keys, color=weight intensity.
4. Aggregate across heads: average, max, or per-head analysis.
5. Compute attention entropy per head/layer; correlate with task performance.

#### Instrumentation & Evaluation
- Attention heatmap grids (layer × head).
- Entropy distributions: box plots per layer/head.
- Token-level attention flow diagrams.
- Attention rollout: cumulative attention across layers.

#### Common Pitfalls & Debugging
- Averaging attention across samples obscures patterns; visualize per-example.
- Causal mask creates triangular patterns; don't mistake for learned behavior.
- Attention weights ≠ importance: high attention doesn't always mean causal influence.
- Over-interpreting single heads: behavior is emergent across all heads.

#### FAQ
- Q: Do all heads learn distinct patterns? A: Often yes, but redundancy exists; some heads are more critical than others.
- Q: Why do models attend to delimiters (e.g., periods, CLS)? A: Aggregation points for sentence-level information.
- Q: Can I prune heads? A: Yes—many heads can be removed with minimal performance loss (model compression strategy).
- Q: How to know which heads matter for my task? A: Ablation studies: mask head, measure task performance drop.

#### Extensions & Next Experiments
- Head pruning: systematically remove heads and measure impact.
- Attention intervention: manually set attention patterns, observe output changes.
- Cross-lingual attention: analyze multilingual models' head behavior.
- BERTology-style probing: which heads capture syntax, semantics, coreference?

#### Alignment to Long-Term Goals
When fine-tuning Mistral, visualizing attention patterns reveals if the model attends to instruction cues properly; attention analysis aids debugging failure modes (e.g., ignoring constraints); understanding head specialization informs which layers to fine-tune (LoRA targeting).

---

## Project 14: Pretraining a Tiny Transformer from Scratch

#### Motivation (Why)
Experience the dynamics of pretraining firsthand: loss, perplexity, scaling behavior, and qualitative sample improvements.

#### Learning Objectives
- Implement a full pretraining loop with checkpoints and sampling.
- Track and interpret loss, perplexity, gradients, and LR schedules.
- Run controlled experiments on LR, capacity, data size, and tokenization.

#### Conceptual Core
- Next-token prediction is maximum-likelihood estimation over sequences.
- Optimization schedules and capacity determine convergence and generalization.

#### Mathematical Foundations
- Loss: [cross-entropy](docs/GLOSSARY.md#cross-entropy-loss) across vocab; [perplexity](docs/GLOSSARY.md#perplexity) = exp(loss).
- [Adam](docs/GLOSSARY.md#adam--adamw) moments (m, v) stabilize noisy [gradients](docs/GLOSSARY.md#gradient).

#### Implementation Steps (How)
1. Assemble model from Project 12; data from 13.
2. Train with Adam; log loss, perplexity; eval on validation.
3. Save checkpoints; generate samples per epoch.

#### Instrumentation & Evaluation
- Scalar logs: loss, ppl, LR, grad norms.
- Qualitative: fixed-prompt generations across epochs.
- Optional: attention maps snapshot over time.

#### Common Pitfalls & Debugging
- Exploding gradients: clip or lower LR.
- Data/tokenizer mismatch causing index errors.
- Overfitting tiny corpora: monitor val loss and early stop.

#### FAQ
- Q: Why does loss stall? A: [LR](docs/GLOSSARY.md#learning-rate-lr) too low/high, [batch](docs/GLOSSARY.md#batch) too small, model under/over-capacity.
- Q: What's a good [perplexity](docs/GLOSSARY.md#perplexity)? A: Relative to corpus/tokenizer; focus on deltas and [validation](docs/GLOSSARY.md#validation-set) gap.
- Q: Do I need LR [warmup](docs/GLOSSARY.md#warmup)/decay? A: Helpful for stability; try cosine with warmup.

#### Alignment to Long-Term Goals
This is the conceptual twin of large-scale pretraining; you’ll interpret fine-tuning behavior through this lens.

---

## Project 15: Analysis — Pretrained vs Random Initialization

#### Motivation (Why)
Quantify the value of pretraining via downstream transfer—evidence over intuition.

#### Learning Objectives
- Compare pretrained vs randomly initialized models on a downstream task.
- Measure sample efficiency, convergence speed, and final performance.

#### Conceptual Core
- Representations learned during pretraining provide a strong prior.
- Fine-tuning adapts representations with fewer updates.

#### Implementation Steps (How)
1. Choose a small supervised task (classification or next-word on new domain).
2. Fine-tune both models with identical pipelines.
3. Plot performance vs training steps; compare curves.

#### Instrumentation & Evaluation
- Learning curves; AUC/accuracy vs steps.
- Data efficiency: performance at small sample counts.

#### Common Pitfalls & Debugging
- Unfair comparisons (different seeds/hparams).
- Data leakage between pretraining and downstream eval.

#### FAQ
- Q: Why does the random model sometimes catch up? A: With enough data/compute; pretraining advantage is most visible in low-data regimes.
- Q: Does pretraining always help? A: Generally yes, but domain shift can reduce gains; adaptation matters.

#### Alignment to Long-Term Goals
Builds the intuition for why we adapt Mistral instead of training from scratch and which regimes benefit most.

---

## PART 3: INSTRUCTION TUNING AND FINE-TUNING

### Bridge Understanding: From Pretraining to Fine-tuning

By now you understand:
- **Pretraining**: Train full model on massive text, learn language patterns
- **Base model**: Result of pretraining with learned weights
- **Fine-tuning**: Take base model, adapt to specific task with small data

Fine-tuning Mistral will feel like Projects 14-15 but simpler:
- You're not updating all parameters (LoRA restricts to low-rank)
- You're on a curated dataset (instructions, not raw text)
- You're specializing existing knowledge, not learning from scratch

---

## Project 16: Instruction Tuning Mistral 7B

**Goal**: Fine-tune the real Mistral 7B base model using knowledge from pretraining.

**What to Build**:
- Load Mistral 7B base model
- Prepare instruction dataset (~10k examples)
- Set up LoRA (rank 32, using concepts from Project 14)
- Training loop (same fundamentals as pretraining, but constrained)
- Evaluate before/after

**Key Concepts Now Clear**:
- The model inside Mistral: you've now built and trained a tiny version
- Attention patterns: you've visualized them
- Token prediction loss: you've optimized it for hours
- Learning dynamics: you've seen loss curves
- Gradient descent: you've done it hundreds of times

**New Wrinkles Specific to Fine-tuning**:
- [LoRA](docs/GLOSSARY.md#lora-low-rank-adaptation): constraining updates to low-rank (more efficient than full pretraining)
- [Instruction formatting](docs/GLOSSARY.md#instruction-tuning): how to structure data
- Evaluation metrics: measuring instruction-following quality
- Safety considerations: does tuning degrade other capabilities? ([catastrophic forgetting](docs/GLOSSARY.md#catastrophic-forgetting))

**Instrumentation** (sophisticated now):
- Compare attention patterns before/after fine-tuning
- Analyze which LoRA parameters change most
- Evaluate on both instruction-following and general language tasks
- Systematically test safety properties (your research interest)

**Why This Is Different Now**: You're not learning what fine-tuning is—you already did that. You're applying it purposefully, analyzing it rigorously.

**Timeline**: 2-4 weeks (including experimentation)

---

## Project 17: Comparative Analysis - Base vs Instruction-Tuned

**Goal**: Systematically compare Mistral base vs your tuned version. This is where your research instincts come in.

**What to Analyze**:
- Generate responses to diverse prompts: both versions
- Compare instruction-following: which does better?
- Compare general capabilities: does tuning hurt general language use?
- Analyze failure modes: where does each fail?
- Attention pattern differences: how does fine-tuning change what the model focuses on?

**Why This Matters**: This is the kind of analysis you'd want to do for your AI safety research. Understanding how fine-tuning changes model behavior is valuable.

**Potential Experiments**:
- Test on pseudoscientific requests (from your earlier research)
- Compare epistemic confidence before/after tuning
- Analyze how instruction-tuning affects reasoning
- Document edge cases

**Timeline**: 2-3 weeks

---

## Complete Learning Timeline

### Phase 1: Classical ML (8-12 weeks)
- Projects 1-11: Gradient descent, algorithms, evaluation

### Phase 2: Transformers & Pretraining (4-6 weeks)
- Project 12: Build transformer from scratch
- Project 13: Tokenization and text preprocessing
- Project 14: Pretrain tiny model (this is the centerpiece)
- Project 15: Analyze pretrained vs random

### Phase 3: Real LLM Fine-tuning (4-6 weeks)
- Project 16: Instruction tune Mistral 7B
- Project 17: Comparative analysis and research

**Total Timeline**: 16-24 weeks (~4-6 months) of engaged learning

This is not fast, but it builds genuine understanding layer by layer.

---

## Concrete M4 Schedule Example

If you committed ~20 hours/week:

**Week 1-10**: Classical ML (Projects 1-11)
- ~2 hours/week reading/learning
- ~18 hours/week hands-on coding and experimentation

**Week 11-14**: Transformers (Projects 12-13)
- Week 11: Understand attention, implement from scratch
- Week 12: Build full transformer
- Week 13: Tokenization and data pipeline
- Week 14: Start Project 14 pretraining runs

**Week 15-16**: Pretraining (Project 14 intensive)
- Day 1: Final setup and first run (4-12 hour training)
- Day 2: Analyze, run variants
- Day 3: Run with different hyperparameters
- Days 4-7: Analysis, experiments, write-up
- Project 15: Comparative analysis

**Week 17-20**: Mistral Fine-tuning (Projects 16-17)
- Week 17: Set up, prepare data
- Week 18: First fine-tuning runs and evaluation
- Week 19: Experimentation and iteration
- Week 20: Deep analysis and research

---

## Key Principles Throughout

**Document Everything**: 
- You're not just learning—you're conducting systematic research
- Keep notebooks with outputs, observations, questions
- Your WhisperEngine analysis skills apply here
- Log loss curves, model outputs, findings

**Instrument Heavily**:
- Every experiment should log what you learn
- Visualize before drawing conclusions
- Compare variants systematically

**Build Intuition Through Experimentation**:
- Change one variable, observe effects
- Don't just follow tutorials—deviate and learn
- Break things intentionally to see failure modes

**Connect to Your Research Interests**:
- Use this as a vehicle for understanding AI safety properties
- Analyze how models respond to edge cases
- Document epistemically interesting behaviors

---

## What You'll Understand After Completing This

**Technical Understanding**:
- How gradient descent works from first principles
- What a transformer is and why it works
- How pretraining creates language understanding
- Why fine-tuning is efficient and effective
- How to evaluate and analyze models rigorously

**Practical Skills**:
- Implement ML/DL from scratch
- Read and debug model code
- Design and run experiments
- Analyze model behavior systematically
- Optimize for your hardware

**Research Capacity**:
- Understand LLM safety in depth
- Analyze AI behavior patterns methodically
- Design experiments to test hypotheses
- Document findings rigorously

**Most Importantly**: You'll move from "I'm using an LLM library" to "I understand what's happening inside this model and how to adapt it purposefully." That's the difference between using tools and mastering them.

---

## Hardware Reality Check for M4

Your setup handles all of this:

| Task | Time | Memory |
|------|------|--------|
| Classical ML projects | Hours to days | <5GB |
| Building transformer | Hours | <1GB |
| Tokenization | Minutes | <100MB |
| Pretrain tiny model (5MB data) | 4-12 hours | ~3GB |
| Fine-tune Mistral 7B | Hours per run | ~20-30GB |
| Analysis/inference | Minutes | ~15GB |

You have headroom for everything. The longest single operation (Mistral fine-tuning) uses maybe 30GB at peak, leaving 34GB for OS and other processes.

---

## Getting Started

1. **Complete classical ML first** (Projects 1-11)
   - Don't skip this
   - It's the foundation everything else builds on
   
2. **Build the transformer** (Project 12)
   - Start small, understand each component
   - Implement attention deeply

3. **Understand tokenization** (Project 13)
   - Seems simple, actually subtle
   - Important for debugging later

4. **Pretrain a tiny model** (Project 14)
   - This is where understanding crystallizes
   - Run multiple variants
   - Obsess over the loss curves and generated text

5. **Do comparative analysis** (Project 15)
   - Rigorously compare pretrained vs random
   - Solidify why pretraining matters

6. **Fine-tune Mistral** (Projects 16-17)
   - Now it will be clear why this works
   - Apply your analysis skills
   - Connect to your research interests

---

## Final Note

This is a substantial commitment, but the payoff is genuine mastery. You'll move from "Claude can do this" to "I understand how this works and can analyze it rigorously." That's a qualitatively different level of knowledge, especially valuable for your AI safety research.

The pretraining section (Project 14) is the lynchpin—that's where classical ML understanding suddenly connects to LLMs. Everything before builds to it, everything after applies it.

Your M4 gives you a real research workstation. Use it.
