# ML → LLM Learning Track Overview

## Purpose
A cohesive, end-to-end curriculum that transitions you from first‑principles classical machine learning to confident, instrumented fine‑tuning of large language models. This is not a shortcut path—it is a depth path. Every stage answers the **why** behind the **how** so later complexity feels inevitable rather than mysterious.

## Philosophy
| Principle | Meaning | Result |
|-----------|---------|--------|
| Build from primitives | Start with linear regression gradients before transformers | Durable intuition |
| Instrument everything | Prefer plots, logs, deltas over assumptions | Early detection of issues |
| Compare & ablate | Change one variable at a time | Clear causal narratives |
| Treat experiments as research | Reproducibility + documentation | Transferable skillset |
| Prioritize clarity over speed | Don’t skip conceptual digestion | Long-term acceleration |

## Competency Milestones
Milestone | Evidence | Projects
---------|----------|---------
Optimization Intuition | Can tune LR, detect divergence | 1–4
Generalization Literacy | Diagnose over vs underfit | 4, 7, 8
Evaluation Rigor | Uses proper metrics & CV | 7, 8, 11
Representation Insight | Engineers features & interprets embeddings | 10, 12–13
Architecture Understanding | Implements transformer block | 12
Pretraining Dynamics | Explains loss, perplexity, scaling | 14
Transfer & Adaptation | Quantifies pretrained vs random | 15
LLM Tuning Mastery | Runs LoRA vs full fine-tune w/ analysis | 16
Behavioral Analysis | Compares pre/post instruction outputs | 17

## Mapping Projects to Long-Term Goals
Long-Term Goal | Supporting Concepts | Related Projects
---------------|---------------------|-----------------
Efficient Fine-Tuning | Regularization, low-rank adaptation | 4, 14–16
Safety & Alignment Analysis | Metrics, evaluation protocol design | 7, 16–17
Model Interpretability | Attention, feature importance, LoRA deltas | 5, 6, 12, 16
Robust Experimentation | Cross-validation, logging, ablations | 8, 11, 14
Architecture Extension | Transformer internals, scaling | 12–14

## Learning Flow (Condensed)
Phase | Focus | Mental Shift
------|-------|-------------
Classical ML (1–11) | Optimization & evaluation fundamentals | "I can reason about loss surfaces." |
Transformer Construction (12–13) | Architecture + data interface | "I know what a block actually computes." |
Tiny Pretraining (14–15) | Dynamics & transfer | "I’ve watched a model learn language." |
Instruction Tuning (16–17) | Adaptation & behavioral analysis | "I can specialize and measure effects." |

## Core Mental Models
Model Behavior Lens | Description | Later Analogy
--------------------|------------|--------------
Loss Landscape | Surface navigated by gradients | High-dim LLM training curvature
Decision Boundary | Separation logic of classifiers | Token probability partitions
Regularization | Constraint shaping capacity | LoRA rank and structural limits
Representation Space | Feature transformations | Embeddings & attention outputs
Ensembling | Variance reduction via aggregation | Mixture-of-experts, adapter fusion

## Documentation Expectations
Every substantial experiment should capture:
1. Configuration (hyperparameters, seed, data slice)
2. Code commit or notebook hash
3. Metrics (loss, accuracy, perplexity, etc.)
4. Visualizations (loss curves, attention maps, decision boundaries)
5. Interpretation (what changed, hypothesized causes)
6. Next action (planned adjustment or closure)

Reuse `PROGRESS_LOG.md` for weekly summaries; add a short appendix in major notebooks with a “Findings & Reflections” section.

## Suggested Weekly Cadence
Day | Activity | Outcome
----|---------|--------
1 | Review theory & plan experiments | Clarity, hypothesis list
2 | Implement + instrument | Working baseline
3 | Run controlled variations | Comparative dataset
4 | Analyze + document | Insight consolidation
5 | Extension or cleanup | Stability & closure
6–7 | Rest or optional reading | Cognitive recovery

## FAQ (Global)
Q: Why spend weeks on classical ML before transformers?  
A: It builds an internal simulator for optimization and evaluation; without it, transformer behaviors feel opaque and debugging stalls.

Q: Can I skip pretraining my own tiny model?  
A: You could, but witnessing dynamics firsthand dramatically clarifies later fine-tuning decisions and interpretation.

Q: How do I avoid just copying code?  
A: Re-derive formulas, write tiny test cases, and audit tensor shapes—active engagement beats passive review.

Q: What if a project feels repetitive?  
A: Repetition in slightly new contexts cements abstraction; look for nuance (e.g., gradient noise differences, metric sensitivity).

Q: How to balance depth with forward momentum?  
A: Define a “minimum insight threshold” per project—once you can explain core trade-offs verbally, move on.

Q: When to re-run an experiment?  
A: If results clash with theory, instrumentation is incomplete, or you cannot yet articulate causes of observed behavior.

Q: Best way to internalize math?  
A: Translate equations into code line-by-line; print intermediate tensors; verify invariants.

Q: How do I measure improvement in intuition?  
A: Attempt to predict qualitative changes (e.g., loss curve shape) before running; compare expectation vs reality.

## Extension Tracks (Post Core Path)
Track | Focus | Sample Extensions
------|-------|------------------
Efficiency | Quantization, distillation | Quantize tiny transformer; compare outputs
Alignment | Safety evaluations | Red-team tuned model; build refusal classifiers
Interpretability | Attention / attribution | Head ablation; LoRA delta visualization
Robustness | Adversarial prompts | Stress-test instruction adherence under perturbations
Evaluation Science | Metric design | Construct composite instruction quality score

## Reading List (Minimal, High-Impact)
Category | Reference | Reason
---------|----------|-------
Optimization | "A Gentle Intro to Optimization" blog / Goodfellow Ch. 8 | Gradient intuition
Regularization | Original Ridge/Lasso papers (skim) | Historical framing
Transformers | Vaswani et al. (2017) | Architectural blueprint
Scaling Laws | Hoffmann et al. (Chinchilla) | Data vs model trade-offs
LoRA | Hu et al. (2021) | Low-rank adaptation mechanism
Alignment | Anthropic / OpenAI alignment overviews | Safety context

## Completion Signal
You are ready to branch into specialized research or advanced production work when you can:
- Accurately diagnose unstable training within 5 minutes of logs/plots.
- Explain LoRA efficacy and limits without notes.
- Design an evaluation suite for a new instruction tuning task from scratch.
- Predict qualitative differences between base and tuned generations before sampling.
- Produce a reproducible experiment report others can replicate.

## Final Advice
Mastery compounds: early diligence in fundamentals pays exponential dividends later. When something feels confusing at scale, build a smaller version and instrument it until it becomes obvious. Curiosity + structured experimentation is the engine of durable expertise.

Proceed with depth, not haste.
