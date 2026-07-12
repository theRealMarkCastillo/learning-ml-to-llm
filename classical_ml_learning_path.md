# Classical ML Learning Foundation
## Prerequisite Learning Path Before LLM/Transformer Training

### Overview
This learning path builds foundational ML concepts from first principles before moving to deep learning and LLMs. Each project teaches core concepts you'll encounter again when training Qwen2.5-1.5B-Instruct, but without architectural complexity.

Why this order: Understanding optimization, loss landscapes, generalization, and validation deeply on simple problems makes LLM training concepts click. You'll know why each step matters, not just follow a script.

What follows turns this into a complete reference: every project includes the why (motivation), what (objectives), how (steps), math, instrumentation, pitfalls, FAQs, and how the concept connects to long-term goals in the learning track.

Note on terminology: If a term is unfamiliar (e.g., cross-entropy, softmax, LoRA, perplexity), see the Glossary at docs/GLOSSARY.md for concise developer-oriented definitions.

---

## Phase 1: Fundamentals from Scratch

### Project 1: Linear Regression from First Principles

#### Motivation (Why)
Linear regression is the smallest playground where optimization, loss, gradients, and convergence are fully visible. If you can reason about these on a line, you can reason about them in high dimensions for transformers.

#### Learning Objectives
- Implement gradient descent and MSE from scratch.
- Build intuition for learning rate and convergence behavior.
- Diagnose training via loss curves and parameter trajectories.

#### Conceptual Core
- Loss as a measure of fit; MSE’s convex landscape makes it ideal to study.
- Gradient points in steepest ascent; negative gradient drives descent.
- Learning rate sets step size along the gradient.

#### Mathematical Foundations
- Hypothesis: ŷ = w x + b
- Loss: [Mean Squared Error (MSE)](docs/GLOSSARY.md#mean-squared-error-mse) = (1/N) Σ (y − ŷ)^2
- Gradients: ∂MSE/∂w = −2/N Σ x (y − ŷ), ∂MSE/∂b = −2/N Σ (y − ŷ)

#### Implementation Steps (How)
1. Generate noisy linear data.
2. Initialize w, b; loop: compute predictions → loss → [gradients](docs/GLOSSARY.md#gradient) → update via [gradient descent](docs/GLOSSARY.md#gradient-descent).
3. Track and plot loss; optionally plot parameter path in (w, b) space.

#### Instrumentation & Evaluation
- Plot loss vs iterations (try linear and log scale).
- Compare runs with different [learning rates](docs/GLOSSARY.md#learning-rate-lr).
- Visualize fitted line against data.

#### Common Pitfalls & Debugging
- Divergence from too-large LR; try reducing by 10x.
- Flat loss: check gradient formula signs and broadcasting.
- Overfitting isn’t typical here; focus on convergence.

#### FAQ
- Q: Why MSE instead of MAE? A: MSE is smooth and differentiable everywhere, simplifying [gradient descent](docs/GLOSSARY.md#gradient-descent) analysis; MAE’s kink at 0 complicates gradients.
- Q: How do I pick [learning rate](docs/GLOSSARY.md#learning-rate-lr)? A: Start with 1e-3 to 1e-1; sweep logarithmically and inspect loss curves.
- Q: Do I need feature scaling? A: For 1D synthetic data, not critical; for multidimensional, [standardization](docs/GLOSSARY.md#standardization) stabilizes optimization.

#### Extensions & Next Experiments
- Add momentum; compare to plain GD.
- Try stochastic/mini-batch updates and observe noise in loss.
- Fit polynomial regression via linear basis expansion to see under/overfit.

#### Alignment to Long-Term Goals
The exact gradient descent mechanics reappear when updating LoRA adapters or full transformer weights; the mental model of “loss → gradient → update” is identical.

---

### Project 2: Binary Classification with Logistic Regression

#### Motivation (Why)
Classification introduces probability outputs and cross-entropy—core to token prediction in LLMs.

#### Learning Objectives
- Implement sigmoid, cross-entropy, and logistic regression training.
- Interpret decision boundaries and probability calibration.
- Understand why we use different losses for regression vs classification.

#### Conceptual Core
- Sigmoid maps logits to probabilities in [0, 1].
- Cross-entropy measures the divergence between predicted probabilities and true labels.
- Gradients derive from log-likelihood of Bernoulli outcomes.

#### Mathematical Foundations
- ŷ = [sigmoid](docs/GLOSSARY.md#sigmoid)(z) with z = w·x + b.
- Binary [cross-entropy](docs/GLOSSARY.md#cross-entropy-loss): −(y log ŷ + (1−y) log(1−ŷ)).
- Gradient of BCE with sigmoid simplifies to (ŷ − y) x.

#### Implementation Steps (How)
1. Generate 2D separable and overlapping datasets.
2. Train via GD on BCE; monitor loss and accuracy.
3. Visualize decision boundary and probability contours.

#### Instrumentation & Evaluation
- Loss curve, accuracy over [epochs](docs/GLOSSARY.md#epoch).
- Calibration plots (optional): predicted prob vs empirical.
- [Confusion matrix](docs/GLOSSARY.md#confusion-matrix) on train/test.

#### Common Pitfalls & Debugging
- Numerical instability: clamp ŷ into [ε, 1−ε] in BCE.
- Class imbalance: accuracy can mislead; use precision/recall.
- Poor boundary: check feature scaling and learning rate.

#### FAQ
- Q: Why not use MSE for classification? A: [Cross-entropy loss](docs/GLOSSARY.md#cross-entropy-loss) aligns with likelihood of Bernoulli outcomes and yields better gradient signals for probabilities.
- Q: Is sigmoid saturation a problem? A: Yes at large |z|; use sensible initialization and scaling.
- Q: How to handle imbalance? A: Class weights, threshold tuning, or resampling.

#### Extensions & Next Experiments
- Add L2 regularization; visualize boundary changes.
- Compare SGD vs full-batch GD noise properties.
- Try non-linear features to capture curved boundaries.

#### Alignment to Long-Term Goals
Cross-entropy and probabilities directly map to next-token prediction in transformers; thresholds mirror decoding choices.

---

### Project 3: Multi-class Classification with Softmax

#### Motivation (Why)
Generalizing to many classes mirrors LLM vocabulary prediction (tens of thousands of classes).

#### Learning Objectives
- Implement softmax regression and one-hot targets.
- Interpret probability distributions across classes.
- Analyze confusion across similar classes.

#### Conceptual Core
- Softmax normalizes logits to a categorical distribution.
- Cross-entropy extends naturally to multi-class.
- Decision boundaries partition space among classes.

#### Mathematical Foundations
- [softmax](docs/GLOSSARY.md#softmax)(z_i) = exp(z_i)/Σ_j exp(z_j).
- Loss: −Σ_k y_k log p_k with [one-hot encoding](docs/GLOSSARY.md#one-hot-encoding) y.
- Gradient: ∂L/∂z = p − y.

#### Implementation Steps (How)
1. Use iris or synthetic 3-class data.
2. Train softmax regression with GD.
3. Plot decision regions and per-class metrics.

#### Instrumentation & Evaluation
- Per-class accuracy, confusion matrix.
- Probability histograms per class.
- Loss/accuracy over time by class.

#### Common Pitfalls & Debugging
- Numerical overflow in exp: subtract max logit.
- Class overlap: expect confusion; examine features.
- Poor convergence: scale features, tune LR.

#### FAQ
- Q: Why one-hot encoding? A: It represents categorical targets for cross-entropy; label indices are insufficient for gradient computation.
- Q: When does softmax fail? A: Non-linear separability; consider adding features or non-linear models.
- Q: How to interpret probabilities? A: They reflect model belief given features; calibration can be checked with reliability curves.

#### Extensions & Next Experiments
- Add polynomial features to improve separability.
- Compare with small neural net (one hidden layer).
- Analyze entropy of predictions across classes.

#### Alignment to Long-Term Goals
Token prediction in LLMs is softmax over vocabulary; this project builds the exact probabilistic lens you’ll reuse.



---



### Project 5: Decision Trees and Feature Importance

#### Motivation (Why)
Trees expose hierarchical, rule-based learning and interpretability—useful contrasts to linear models and neural nets.

#### Learning Objectives
- Train and visualize decision trees; analyze splits.
- Understand overfitting via depth and pruning.
- Interpret impurity-based feature importance.

#### Conceptual Core
- Greedy splitting optimizes local impurity reduction.
- Depth increases flexibility at risk of variance.
- Importance reflects average impurity decrease contributions.

#### Mathematical Foundations
- Impurity: Gini or entropy; information gain = Δimpurity.
- Stopping criteria: min samples, max depth, min improvement.

#### Implementation Steps (How)
1. Train trees with varying depths.
2. Visualize structure and decision regions.
3. Compare to logistic regression on same data.

#### Instrumentation & Evaluation
- Plot decision regions; overlay misclassified points.
- Track validation accuracy vs depth.
- Feature importance bar plots.

#### Common Pitfalls & Debugging
- High-variance trees: use validation to set depth.
- Misinterpreting importance: correlated features can mislead.
- Data leakage in preprocessing steps.

#### FAQ
- Q: Why do trees overfit? A: They can memorize small regions; regularize with depth/pruning/min samples.
- Q: Gini vs entropy? A: Often similar; entropy is information-theoretic, Gini is computationally cheaper.
- Q: Are feature importances causal? A: No—they’re correlational and dataset-dependent.

#### Extensions & Next Experiments
- Cost-complexity pruning experiments.
- Permutation importance (model-agnostic).
- Partial dependence plots for interpretability.

#### Alignment to Long-Term Goals
Hierarchical pattern learning echoes deeper transformer layers’ abstraction building; interpretability tools inform LLM behavior analysis.

---

### Project 6: Ensemble Methods (Random Forests)

#### Motivation (Why)
Ensembles reduce variance and improve robustness by aggregating weak learners—an idea echoed by mixture-of-experts.

#### Learning Objectives
- Train random forests; understand bagging and bootstrap.
- Analyze error vs number of trees.
- Compare stability of feature importances.

#### Conceptual Core
- Averaging decorrelated models reduces variance.
- Bootstrap sampling and feature subsampling create diversity.
- Law of large numbers underpins performance gains.

#### Mathematical Foundations
- Bias-variance decomposition; variance reduction via averaging.
- Out-of-bag error as an internal validation estimate.

#### Implementation Steps (How)
1. Train forests with different n_estimators and max_features.
2. Track OOB error vs trees.
3. Compare to single tree performance.

#### Instrumentation & Evaluation
- Error curve vs number of trees.
- Distribution of predictions across trees.
- Stability of importances across random seeds.

#### Common Pitfalls & Debugging
- Diminishing returns after certain tree counts.
- Correlated trees reduce benefit—use subsampling effectively.
- Longer training times; profile and batch wisely.

#### FAQ
- Q: Why do ensembles help if base learners are biased? A: They primarily reduce variance; if bias is high, consider boosting.
- Q: How many trees are enough? A: Until OOB/validation error plateaus; often hundreds suffice.
- Q: Are forests interpretable? A: Less than a single tree; use permutation importance/SHAP.

#### Extensions & Next Experiments
- ExtraTrees (more randomness) vs RandomForest.
- Gradient boosting comparison (XGBoost/LightGBM).
- Calibrate probabilities (isotonic/platt scaling).

#### Alignment to Long-Term Goals
Mixture-of-experts ensembling ideas appear in modern LLMs; understanding variance reduction aids system-level design.





---



#### Motivation (Why)
Sound experimental design prevents overconfident results and supports reproducible conclusions.

#### Learning Objectives
- Implement k-fold, [stratified k-fold](docs/GLOSSARY.md#stratified-k-fold), and [time-series CV](docs/GLOSSARY.md#time-series-cross-validation).
- Report mean ± std across folds; detect instability.
- Use CV for hyperparameter selection.

#### Conceptual Core
- Variance across folds quantifies uncertainty.
- Stratification preserves label distribution.
- [Leakage](docs/GLOSSARY.md#data-leakage) control is paramount (fit transforms inside folds only).

#### Mathematical Foundations
- Estimators of mean performance and its variance.
- Bias correction considerations for small k.

#### Implementation Steps (How)
1. Build CV pipelines with proper transformers inside folds.
2. Grid search hyperparameters with CV.
3. Compare single split vs k-fold outcomes.

#### Instrumentation & Evaluation
- Fold-wise results table and variance bar plot.
- Learning curves vs data size per fold.
- Sensitivity analysis to random seeds.

#### Common Pitfalls & Debugging
- Leakage: scaling/feature selection outside folds.
- Data dependence violations in time series; use blocked CV.
- Overfitting CV by excessive hyperparameter search.

#### FAQ
- Q: How to choose k? A: 5 or 10 are common; trade-off compute with variance of estimate.
- Q: Do I need a final test set? A: Yes—reserve untouched data for final reporting.
- Q: Is nested CV necessary? A: For small data and heavy tuning, yes to avoid optimistic bias.

#### Extensions & Next Experiments
- Nested CV; repeated stratified k-folds.
- Grouped CV for grouped observations.
- Learning curve experiments to estimate data needs.

#### Alignment to Long-Term Goals
Fine-tuning experiments need rigorous validation to compare setups fairly and avoid overfitting to a dev set.

---

### Project 9: Support Vector Machines

#### Motivation (Why)
Margins and kernels provide a different perspective on separating data and controlling complexity.

#### Learning Objectives
- Visualize margins and support vectors; tune C and γ.
- Understand kernel trick intuition.
- Compare linear vs RBF performance and boundaries.

#### Conceptual Core
- Maximum-margin principle; only support vectors matter for boundary.
- Kernels implicitly map to high-dimensional spaces.

#### Mathematical Foundations
- Primal/dual formulations; [hinge loss](docs/GLOSSARY.md#hinge-loss); role of C.
- [RBF kernel](docs/GLOSSARY.md#rbf-kernel): exp(−γ||x−x′||²).

#### Implementation Steps (How)
1. Train linear and RBF [SVMs](docs/GLOSSARY.md#support-vector-machine-svm); visualize.
2. Tune C and γ via CV; inspect over/underfit.
3. Highlight support vectors on plots.

#### Instrumentation & Evaluation
- Margin width visualization.
- Decision function heatmaps.
- Grid of C, γ vs validation score.

#### Common Pitfalls & Debugging
- Extreme C leads to overfit; too small underfits.
- γ too high produces noisy boundaries.
- Scaling features is essential for RBF.

#### FAQ
- Q: Why do only some points matter? A: The KKT conditions select support vectors that define the boundary.
- Q: Is soft margin better than hard? A: Typically yes—robust to noise/outliers.
- Q: Can I interpret SVMs? A: Linear SVMs are interpretable via weights; kernels less so.

#### Extensions & Next Experiments
- One-vs-rest vs one-vs-one for multi-class SVMs.
- Compare with logistic regression on same features.
- Try polynomial kernels and compare.

#### Alignment to Long-Term Goals
Margin-based thinking and kernel intuition enrich your mental models for representation learning and loss shaping.

---

### Project 10: Feature Engineering and Representation

#### Motivation (Why)
Representation quality often outweighs algorithm choice; deep learning’s power comes from learned representations.

#### Learning Objectives
- Engineer polynomial and [interaction features](docs/GLOSSARY.md#interaction-feature).
- Apply [standardization](docs/GLOSSARY.md#standardization)/[normalization](docs/GLOSSARY.md#normalization) and understand effects on algorithms.
- Quantify gains from representation changes.

#### Conceptual Core
- Features shape the hypothesis space accessible to a model.
- Scaling stabilizes gradient-based optimization and distance-based models.

#### Mathematical Foundations
- Standardization: (x−μ)/σ; Normalization: x/||x||.
- Polynomial feature expansion increases effective capacity linearly.

#### Implementation Steps (How)
1. Baseline model on raw features.
2. Add engineered features; retrain and compare.
3. Apply scaling; compare convergence and accuracy.

#### Instrumentation & Evaluation
- Ablation table: raw vs engineered vs scaled.
- Convergence speed comparison (iterations to tolerance).
- Feature importance/coefficients analysis.

#### Common Pitfalls & Debugging
- Leakage: fit scalers on train only.
- Too many features → overfitting; regularize or select.
- Interpretability vs performance trade-offs.

#### FAQ
- Q: Which models need scaling? A: Most gradient/distance-based (LR, SVM, KNN); trees less sensitive.
- Q: Are polynomial features obsolete with DL? A: Not on small tabular problems; they’re simple and effective.
- Q: How to pick features? A: Domain knowledge + systematic ablations.

#### Extensions & Next Experiments
- Feature selection (L1, mutual information).
- Non-linear kernels vs explicit features.
- PCA for dimensionality reduction.

#### Alignment to Long-Term Goals
Relates to embeddings in LLMs—learned features outperform manual ones at scale; you’ll recognize the role embeddings play.

---

### Project 11: End-to-End ML Pipeline

#### Motivation (Why)
Real-world problems require integrating data, modeling, validation, and documentation into a coherent workflow.

#### Learning Objectives
- Build a reproducible pipeline with clear experiments.
- Use proper validation and reporting.
- Make principled decisions grounded in evidence.

#### Conceptual Core
- Iterative loop: explore → hypothesize → experiment → evaluate → decide.
- Reproducibility through fixed seeds and environment capture.

#### Implementation Steps (How)
1. Select dataset; define problem and metrics.
2. Baseline → iterate with features/models.
3. Cross-validate, document, and present findings.

#### Instrumentation & Evaluation
- Experiment logs with configs and results.
- Final report with metrics and error analysis.
- Reproducible scripts/notebooks.

#### Common Pitfalls & Debugging
- Moving goalposts during iteration.
- Hidden leakage in preprocessing.
- Cherry-picking best runs without validation discipline.

#### FAQ
- Q: How much documentation is enough? A: Enough to recreate results and decisions from scratch.
- Q: How many models to try? A: Start simple; iterate purposefully; stop when gains flatten.
- Q: What’s a good baseline? A: A trivial yet transparent model (mean predictor, logistic regression) to calibrate difficulty.

#### Extensions & Next Experiments
- Add automated reporting and plotting.
- Try lightweight model selection frameworks.
- Package the pipeline for reuse.

#### Alignment to Long-Term Goals
This mirrors how you'll run Qwen2.5 fine-tuning: clear goals, solid validation, disciplined experimentation, thorough analysis.

---

## Bridge Projects to Deep Learning

After Project 11, you'll find four **bridge projects** in this repo that prepare you for the transformer content. These are not described in detail in this markdown doc — they live as standalone notebooks with their own intros. Run them in this order:

- **Project 11.5** — Neural Networks from Scratch (NumPy MLP, manual backprop, depth vs width)  
  `projects/phase1_classical_ml/project11_5_neural_networks/neural_networks_from_scratch.ipynb`
- **Project 11.75** — RNNs from Scratch (vanilla RNN + BPTT, vanishing gradients, character-level language modeling on Shakespeare)  
  `projects/phase1_classical_ml/project11_75_rnns/recurrent_neural_networks.ipynb`

Both are short (~2-3 days each) and exist to make the eventual transformer content feel inevitable rather than mysterious. The detailed guides for the Phase 2 bridges (12.1 attention, 12.25 embeddings) live in `complete_ml_learning_path_with_pretraining.md`.

---

## Tools and Environment Setup

### Required Libraries
```bash
pip install numpy scipy scikit-learn pandas matplotlib seaborn jupyter
```

### Suggested Structure
- Create a directory for each project (project1_linear_regression, etc.).
- Keep a Jupyter notebook per project.
- Document findings and learnings in markdown cells.
- Include visualizations inline.

### Development Approach
For each project:
1. Start with theoretical understanding (read/watch conceptual materials).
2. Implement from scratch using NumPy first.
3. Compare against sklearn/library implementations.
4. Visualize everything you can.
5. Experiment with hyperparameters.
6. Document what you learn.

---

## Learning Outcomes

By completing this progression, you'll deeply understand:

- Optimization: how gradient descent works; learning rates; convergence.
- Loss Functions: MSE, cross-entropy, and regularization.
- Generalization: overfitting/underfitting, regularization, validation.
- Model Evaluation: when to use which metrics; proper experimental design.
- Feature Representation: how models learn or engineer features.
- Algorithm Diversity: when different approaches shine.
- Experimental Rigor: how to design, run, and document experiments.

Direct transfer to LLM training:
- Gradient descent → updating LoRA or full weights.
- Cross-entropy loss → token prediction loss.
- Regularization → LoRA and prompt/adapter constraints.
- Validation strategy → evaluating instruction-tuned models.
- Experimentation rigor → systematic fine-tuning analysis.

---

## Progression Timeline Estimate

Working systematically through all projects (not rushing):
- Projects 1-4 (Fundamentals): 2-3 weeks (intensive, foundational).
- Projects 5-7 (Algorithms & Evaluation): 2-3 weeks.
- Projects 8-9 (Advanced Concepts): 1-2 weeks.
- Project 10 (Feature Engineering): 1 week.
- Project 11 (End-to-End): 2-3 weeks.

Total: 8-12 weeks before moving to Qwen2.5/LLM training.

---

## Next Steps

1. Start with Project 1; code from scratch and instrument heavily.
2. Document everything—logs, plots, insights.
3. Don’t rush; depth over breadth.
4. After Project 11, proceed to transformers/pretraining with confidence.

Once you finish Project 11, you'll be ready to return to the Qwen2.5-1.5B-Instruct project with solid conceptual foundations.
