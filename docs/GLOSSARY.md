# Glossary of Core ML & LLM Terms

Audience: Experienced software engineer new to ML/LLM training. Each term includes: (Definition) – Why it matters – Developer mental model.

## A
### Activation Function
Non-linear transform applied to layer outputs (e.g., ReLU, GELU). Prevents model collapsing into linear mapping. Think: pluggable math shaping information flow.

### Adapter (LoRA Adapter)
Small trainable module injected into a frozen large model to enable efficient fine-tuning. Think: patch layer that learns deltas.

### Adam / AdamW
Adaptive optimizers maintaining first/second moment estimates of gradients (AdamW decouples weight decay). Stabilizes and speeds convergence compared to plain SGD.

### AUC (Area Under ROC Curve)
Probability a classifier ranks a random positive above a random negative. Measures ranking quality independent of threshold.

## B
### Backpropagation
Algorithm for computing gradients layer-by-layer via chain rule. Runtime cost ~ forward pass. Think: reverse dataflow pass computing local sensitivities.

### Bagging
Bootstrap aggregating—train multiple models on resampled data and average predictions to reduce variance.

### Batch
Group of samples processed together for gradient estimation. Larger batch → lower gradient noise → higher memory.

### Bias (Parameter)
Additive scalar/vector shifting a linear transformation. Allows non-zero-centered outputs.

### Bias–Variance Trade-off
Balance between underfitting (high bias) and overfitting (high variance). Goal: minimal generalization error.

### BLEU / ROUGE
Text generation overlap metrics (n-gram precision / recall flavored). Crude; use cautiously for instruction models.

### Bootstrap Sampling
Sampling with replacement to create datasets for bagging; enables out-of-bag error estimation.

### Byte Pair Encoding (BPE)
Subword tokenization algorithm merging frequent pairs to compress corpus; balances vocab size and fragmentation.

## C
### Calibration
Alignment between predicted probabilities and true frequencies. Important for reliability under decision thresholds.

### Categorical Cross-Entropy
Loss comparing predicted probability distribution to one-hot true distribution. Core to token prediction.

### Catastrophic Forgetting
Loss of previously learned capabilities during fine-tuning; mitigated by adapters, mixing data, or constraints.

### Checkpoint
Serialized model (and optimizer) state for resume and comparison.

### Confusion Matrix
2D table counting TP/FP/FN/TN; foundation for many metrics.

### Context Window
Maximum sequence length model can see in one forward pass. Transformers attend across this window.

### Cross-Entropy Loss
Measures divergence between target distribution and predicted probabilities. Equivalent to negative log-likelihood.

### Cross-Validation (CV)
Experimental protocol splitting data into folds to estimate generalization. Reduces variance vs single split.

## D
### Data Leakage
Test/validation information inadvertently used during training causing optimistic metrics.

### Decoder-Only Transformer
Architecture predicting next token from all prior tokens (causal mask). Used in GPT/Mistral.

### Decision Boundary
Hypersurface separating predicted classes. Visualization tool for model discriminative behavior.

### Dropout
Regularization technique randomly zeroing activations during training to reduce co-adaptation.

## E
### Early Stopping
Halt training when validation metric stops improving to prevent overfitting. Acts like implicit regularization.

### Embedding
Dense vector representation of discrete tokens. Learned lookup table mapping ids → continuous space.

### Epoch
One full pass over the training dataset.

### Evaluation Suite
Structured set of metrics/tests/prompts to judge model performance and regressions.

## F
### F1 Score
Harmonic mean of precision and recall. Useful when balancing false positives vs false negatives.

### Feed-Forward Network (FFN)
Per-token MLP inside transformer block; expands then contracts dimensionality for non-linear mixing.

### Feature Importance
Score indicating contribution of features to model predictions (tree impurity-based or permutation-based).

### Feature Engineering
Creating or transforming input features to improve model performance (polynomial, interactions, encoding).

### Fine-Tuning
Adapting a pretrained model to a narrower task/distribution using additional data.

### Forward Pass
Computing model outputs from inputs. Precedes loss and gradient computation.

## G
### Generalization
Performance on unseen data. Ultimate goal; distinguishes memorization from learning patterns.

### GELU
Activation approximating Gaussian gating; common in transformers.

### Gradient
Vector of partial derivatives of loss w.r.t. parameters. Direction of steepest local ascent; negated for descent.

### Gradient Clipping
Capping gradient norm to stabilize training under spikes.

### Gradient Descent
Optimization method updating parameters opposite the gradient to minimize loss. Includes variants (batch, SGD, Adam).

## H
### Head (Attention Head)
Independent projection in multi-head attention focusing on different relational patterns.

### Hidden Dimension
Internal feature vector size. Larger → more capacity + compute cost.

### Hinge Loss
Margin-based classification loss (e.g., SVM) encouraging decision boundary spacing.

## I
### Inference
Using a trained model to generate predictions (no gradient updates).

### Instruction Tuning
Fine-tuning an LLM on prompt→response pairs to improve alignment with human-style directives.

### Iteration (Step)
Single parameter update (one batch forward+backward).

### Interaction Feature
Feature formed by combining two or more base features (e.g., product), capturing interactions.

## J
### JIT (Just-In-Time Compilation)
Runtime graph/kernel optimization (not always applicable in MLX yet; concept placeholder).

## K
### Kernel (SVM)
Function implicitly mapping inputs to higher-dimensional space enabling linear separability.

### KL Divergence
Measure of how one probability distribution diverges from another; used in some regularization schemes.

## L
### LayerNorm
Normalization over feature dimension stabilizing activations in transformers.

### Learning Rate (LR)
Step size scaling gradient updates. Critical hyperparameter.

### Learning Rate Schedule
Planned variation of learning rate over training (e.g., warmup, cosine decay) to improve stability and convergence.

### Logit
Raw (pre-softmax) unnormalized score per class/token.

### LoRA (Low-Rank Adaptation)
Technique: factor weight update ΔW into low-rank matrices A·B to reduce trainable parameter count.

### Loss Function
Scalar objective optimized during training; proxies “how wrong” predictions are.

### Low-Rank Constraint
Structural restriction reducing degrees of freedom; acts as regularization.

## M
### Mask (Attention Mask)
Tensor preventing attention to certain positions (e.g., future tokens or padding).

### Mean Squared Error (MSE)
Regression loss: average squared difference between predictions and targets.

### Metric
Quantitative performance measure distinct from optimization objective (e.g., accuracy vs cross-entropy).

### Mini-Batch
Subset of data per iteration for stochastic gradient estimation.

### MLP (Multi-Layer Perceptron)
Sequence of linear layers + activations; building block inside transformers.

### Momentum
Optimizer technique accumulating past gradients to smooth updates.

## N
### Normalization
Rescaling values to stabilize training (e.g., LayerNorm, standardization).

### NumPy
Python numerical array library used for foundational implementations.

## O
### Objective
Same as loss; what optimization minimizes.

### One-Hot Encoding
Vector representation where only the index of the class is 1 and others 0; used with categorical cross-entropy.

### Optimizer
Algorithm updating parameters using gradients (SGD, Adam, AdamW).

### Overfitting
Low training error, high validation error; model memorizes noise.

## P
### Padding
Filling shorter sequences to uniform length; masked to avoid influencing outputs.

### Parameter
Trainable scalar within the model.

### Perplexity
exp(cross-entropy); average branching factor; lower is better for language models.

### Positional Encoding
Mechanism to inject token position information (e.g., sinusoidal, rotary) into transformer inputs.

### Precision (Metric)
TP / (TP + FP); reliability of positive predictions.

### Prompt
Input text guiding LLM generation.

### Pruning
Removing weights/structures to shrink model size.

## Q
### Quantization
Reducing numeric precision (e.g., FP16 → INT8) for speed/memory savings.

### Query / Key / Value
Matrices in attention: query seeks information, key indexes, value supplies content.

## R
### Recall
TP / (TP + FN); coverage of actual positives.

### Regularization
Techniques to reduce overfitting (L1/L2, dropout, early stopping, low-rank constraints).

### Residual Connection
Adds input to block output; eases gradient flow in deep nets.

### ROC Curve
TPR vs FPR across thresholds; summarizes ranking performance.

### Rogue Token (Generation)
Token that derails output style or structure; often mitigated via constrained decoding.

### RBF Kernel
Radial basis function kernel exp(−γ||x−x′||²) enabling flexible non-linear boundaries in SVMs.

## S
### Scaling Laws
Empirical relations between model/data/compute scale and loss.

### Schedule (LR Schedule)
Planned variation of learning rate over training (warmup, cosine decay).

### Seed (Random Seed)
Deterministic initializer for reproducibility of randomness.

### Sigmoid
Activation σ(z)=1/(1+e^{−z}) mapping logits to probabilities in binary classification.

### Softmax
Normalizes logits into probability distribution; exponentiate + normalize.

### Standardization
Feature scaling to zero mean and unit variance: (x−μ)/σ.

### Stochastic Gradient Descent (SGD)
Parameter updates using batch-sampled gradients; adds noise aiding generalization.

### Stratified K-Fold
Cross-validation ensuring each fold preserves label distribution.

### Support Vector
Training sample lying on or within margin influencing SVM boundary.

### Support Vector Machine (SVM)
Classifier maximizing margin; uses kernels to model non-linear boundaries.

## T
### Temperature (Sampling)
Scales logits during generation; higher → more randomness.

### Time Series Cross-Validation
Cross-validation that respects temporal order using expanding/rolling windows.

### Tokenizer
Maps raw text to token IDs and back.

### Training Loop
Iterative process: load batch → forward → loss → backward → update.

### Transfer Learning
Leveraging pretrained representations for downstream tasks.

### Transformer Block
Composition: attention + residual + norm + FFN + residual + norm.

## U
### Underfitting
Both training and validation performance poor; model too simple or undertrained.

### Unfreezing
Allowing previously frozen layers to become trainable.

## V
### Validation Set
Held-out subset guiding hyperparameter tuning and early stopping.

### Variance (Model Variance)
Sensitivity of model predictions to data sampling; high variance → overfitting.

### Vocabulary (Vocab)
Set of all unique tokens recognized by tokenizer.

## W
### Warmup
Period of gradually increasing learning rate to stabilize initial training.

### Weight Decay
L2-like penalty integrated into optimizer; discourages large weights.

## X
### (No common core term)
(Reserved for future extensions.)

## Y
### y / ŷ (Prediction Notation)
y is ground truth; ŷ ("y-hat") is model prediction.

## Z
### Zero-Shot
Model performs task without task-specific fine-tuning; relies on generalization from pretraining.

---
## Cross-Reference Tags
Use these tags in project notes for quick linking:
- #optimization (LR, gradients, convergence)
- #regularization (L1, L2, LoRA rank)
- #evaluation (metrics, calibration, ROC)
- #architecture (transformer, attention)
- #adaptation (fine-tuning, LoRA, transfer)
- #tokenization (BPE, vocab, sequence length)

---
## How to Extend
When you encounter a new term:
1. Write a 1-sentence definition.
2. Add why it matters for your current project.
3. Note one experiment that would clarify it.

---
## Quick Mental Models
Concept | Mental Compression
--------|--------------------
Gradient | Local slope telling you how to tweak code-generated outputs.
Attention | Dynamic content-based join between sequence positions.
LoRA | Low-rank patch overlaying original weights.
Perplexity | Average branching factor; lower = more certainty.
Regularization | Guardrail preventing overconfident memorization.

Keep this glossary open while working through early projects; reinforce by re-defining terms in your own words.
