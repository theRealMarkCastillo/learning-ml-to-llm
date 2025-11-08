# Classical ML Learning Foundation
## Prerequisite Learning Path Before LLM/Transformer Training

### Overview
This learning path builds foundational ML concepts from first principles before moving to deep learning and LLMs. Each project teaches core concepts you'll encounter again when training Mistral 7B, but without architectural complexity.

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

### Project 4: Regularization and Overfitting

#### Motivation (Why)
Generalization—performing well on unseen data—is the whole game; regularization is how we control complexity.

#### Learning Objectives
- Diagnose under/overfitting with train/test curves.
- Implement L1/L2 penalties and interpret effects on parameters.
- Understand bias-variance trade-off empirically.

#### Conceptual Core
- Regularization penalizes complexity to reduce variance.
- L2 shrinks weights smoothly; L1 induces sparsity.
- Model capacity vs data size determines overfitting risk.

#### Mathematical Foundations
- Ridge loss: [MSE](docs/GLOSSARY.md#mean-squared-error-mse) + λ||w||²; Lasso: MSE + λ||w||₁.
- Effect: gradients include penalty terms (2λw for L2; sign(w) for L1).

#### Implementation Steps (How)
1. Fit polynomials of increasing degree to noisy sine.
2. Add L2 and L1; sweep λ.
3. Compare train vs test loss; plot parameter magnitudes.

#### Instrumentation & Evaluation
- Train/test curves on same plot.
- Parameter paths as λ increases.
- Visualize fitted curves vs degree and λ.

#### Common Pitfalls & Debugging
- Over-regularization: high bias; curve too flat.
- Under-regularization: high variance; wild oscillations.
- Leakage: ensure test set is untouched during tuning.

#### FAQ
- Q: When use L1 vs L2? A: L1 for sparsity/feature selection; L2 for smooth shrinkage and numerical stability.
- Q: Is [early stopping](docs/GLOSSARY.md#early-stopping) a regularizer? A: Yes—limits effective capacity by cutting training short.
- Q: How to pick λ? A: [Cross-validation](docs/GLOSSARY.md#cross-validation-cv) on validation folds.

#### Extensions & Next Experiments
- Compare early stopping vs explicit L2.
- Elastic Net (mix of L1/L2) to balance sparsity and stability.
- Add noise levels to study robustness.

#### Alignment to Long-Term Goals
LoRA is a structural regularizer (low-rank constraint). Understanding classic regularizers informs better adapter rank/alpha choices.

---

### Project 4.5: Learning Curves and Model Selection

#### Motivation (Why)
Understanding when more data helps vs when you need a better model is crucial for resource allocation and experimental design.

#### Learning Objectives
- Plot and interpret learning curves (performance vs training set size).
- Diagnose high bias vs high variance from curve shapes.
- Use information criteria (AIC/BIC) for model comparison.
- Implement systematic model selection workflows.

#### Conceptual Core
- Learning curves reveal whether you're data-limited or model-limited.
- High bias: train and test error both high (model too simple).
- High variance: large gap between train and test (model too complex).
- Information criteria balance fit quality with model complexity penalties.

#### Mathematical Foundations
- AIC = 2k − 2ln(L̂), BIC = k·ln(n) − 2ln(L̂) where k = parameters, n = samples, L̂ = likelihood.
- BIC penalizes complexity more heavily than AIC.
- Learning curve: plot E_train and E_test vs m (training size).

#### Implementation Steps (How)
1. Generate dataset; split into train/test.
2. For increasing train sizes: fit model → compute train/test error → plot.
3. Repeat for models of different complexity (linear, polynomial degrees).
4. Compute AIC/BIC for nested models; compare rankings.

#### Instrumentation & Evaluation
- Plot curves with error bars (multiple random splits).
- Annotate bias/variance regimes on plots.
- Table comparing models: complexity, AIC, BIC, test error.

#### Common Pitfalls & Debugging
- Not using fixed test set across train sizes (causes noisy curves).
- Extrapolating beyond observed data range.
- Confusing AIC/BIC: BIC more conservative for large n.

#### FAQ
- Q: When do learning curves plateau? A: When model capacity matches data complexity or you've exhausted useful signal.
- Q: Should I always trust AIC/BIC? A: They're guides assuming certain regularity conditions; validate on held-out data.
- Q: How much data do I need? A: Learning curves tell you if gains are flattening or still rising.

#### Extensions & Next Experiments
- Bayesian model averaging instead of selection.
- PAC learning bounds for theoretical sample complexity.
- Compare nested CV for model selection.

#### Alignment to Long-Term Goals
When fine-tuning LLMs, understanding if poor performance stems from insufficient data vs wrong architecture informs whether to collect more examples or change model capacity (LoRA rank, layers unfrozen).

---

## Phase 2: Standard Algorithms

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

### Project 6.5: K-Nearest Neighbors and Naive Bayes

#### Motivation (Why)
KNN and Naive Bayes represent opposite ends of the ML spectrum: non-parametric instance-based vs parametric probabilistic. Both serve as essential baselines.

#### Learning Objectives
- Implement KNN with different distance metrics and k values.
- Build Naive Bayes classifiers (Gaussian, Multinomial, Bernoulli variants).
- Understand when simple baselines outperform complex models.
- Analyze computational trade-offs (training vs inference).

#### Conceptual Core
- KNN: prediction by local neighborhood voting; no explicit training phase.
- Distance metrics shape neighborhoods; choice matters with feature scales.
- Naive Bayes: assumes feature independence given class; fast and interpretable.
- Prior and likelihood estimates from training data via counting/density estimation.

#### Mathematical Foundations
- KNN: ŷ = mode({y_i : x_i ∈ N_k(x)}) for classification; average for regression.
- Distance: Euclidean, Manhattan, Minkowski; affected by feature scales.
- Naive Bayes: P(y|x) ∝ P(y) ∏_j P(x_j|y) with independence assumption.
- Gaussian NB: P(x_j|y) ~ N(μ_jy, σ²_jy); Multinomial for count data.

#### Implementation Steps (How)
1. Implement KNN from scratch: distance computation, k-neighbor selection, voting.
2. Test on synthetic 2D data with varying k; visualize decision boundaries.
3. Implement Gaussian and Multinomial Naive Bayes.
4. Compare all methods on text classification (bag-of-words) and numeric data.

#### Instrumentation & Evaluation
- KNN: error vs k curves; visualize neighbors for sample points.
- NB: inspect learned priors and likelihoods; plot class-conditional distributions.
- Confusion matrices comparing methods.
- Timing: train time, prediction time per method.

#### Common Pitfalls & Debugging
- KNN without feature scaling: dominated by large-scale features.
- k too small: overfit to noise; k too large: oversmooth.
- Naive Bayes independence violation: check feature correlations; still often works.
- Zero probabilities in NB: use Laplace smoothing.

#### FAQ
- Q: When does KNN fail? A: High dimensions (curse of dimensionality), large datasets (slow inference), irrelevant features.
- Q: Why does Naive Bayes work despite violated assumptions? A: Robust probability ranking even with wrong independence; calibration may suffer.
- Q: Which distance metric for KNN? A: Euclidean for continuous features after scaling; Manhattan more robust to outliers.

#### Extensions & Next Experiments
- Weighted KNN (closer neighbors count more).
- Approximate nearest neighbors for scaling (LSH, ball trees).
- Semi-supervised learning with NB (EM algorithm).
- Compare with logistic regression as probabilistic baseline.

#### Alignment to Long-Term Goals
Understanding baseline model performance sets expectations for complex models; KNN's local reasoning echoes retrieval-augmented generation in LLMs; Naive Bayes' probabilistic reasoning connects to language model likelihood computations.

---

## Phase 3: Evaluation, Validation, and Advanced Methods

### Project 7: Classification Metrics Deep Dive

#### Motivation (Why)
Accuracy can deceive; robust evaluation requires the right metric for the task and costs.

#### Learning Objectives
- Compute and interpret precision, recall, F1, ROC-AUC, PR-AUC.
- Understand threshold effects and calibration.
- Design evaluation protocols for imbalanced data.

#### Conceptual Core
- Confusion matrix as the base object; metrics derive from it.
- Threshold tuning changes confusion trade-offs.
- ROC vs PR focus: separability vs performance on rare positives.

#### Mathematical Foundations
- Precision = TP/(TP+FP); Recall = TP/(TP+FN); F1 = 2PR/(P+R).
- ROC curve from TPR vs FPR at varying thresholds.

#### Implementation Steps (How)
1. Train multiple models on imbalanced data.
2. Plot ROC and PR curves; compute AUCs.
3. Sweep thresholds; produce metric vs threshold plots.

#### Instrumentation & Evaluation
- Heatmap [confusion matrices](docs/GLOSSARY.md#confusion-matrix); annotate errors.
- [Calibration](docs/GLOSSARY.md#calibration) curves: predicted prob vs observed frequency.
- Cost-sensitive summaries (assign costs to FP/FN).

#### Common Pitfalls & Debugging
- Using accuracy on imbalanced sets.
- Ignoring calibration when using probabilities operationally.
- Comparing models at different thresholds unfairly.

#### FAQ
- Q: ROC or PR for rare events? A: PR is more informative with class imbalance.
- Q: How to pick a threshold? A: Optimize for your cost function or [F1](docs/GLOSSARY.md#f1-score); validate on held-out data.
- Q: Why does [AUC](docs/GLOSSARY.md#auc-area-under-roc-curve) look good but precision is low? A: AUC summarizes ranking quality, not operating-point precision.

#### Extensions & Next Experiments
- Cost curves and decision analysis.
- Expected calibration error (ECE) computation.
- Bootstrapped confidence intervals for metrics.

#### Alignment to Long-Term Goals
You'll evaluate instruction-tuned models along multiple axes; this rigor avoids misleading conclusions.

---

### Project 7.5: Clustering and Dimensionality Reduction

#### Motivation (Why)
Unsupervised learning reveals data structure without labels—critical for exploratory analysis, preprocessing, and understanding representations.

#### Learning Objectives
- Implement K-means clustering from scratch; understand convergence.
- Apply PCA for visualization and noise reduction.
- Compare clustering quality metrics (silhouette, inertia, elbow method).
- Visualize high-dimensional embeddings in 2D/3D.

#### Conceptual Core
- K-means: iterative assignment and centroid update minimizes within-cluster variance.
- Initialization matters (k-means++ improves convergence).
- PCA: finds orthogonal directions of maximum variance via eigendecomposition.
- Dimensionality reduction trades information loss for interpretability.

#### Mathematical Foundations
- K-means objective: minimize Σ_k Σ_{x∈C_k} ||x − μ_k||².
- Lloyd's algorithm: E-step (assign) → M-step (update centroids) → repeat.
- PCA: eigenvectors of covariance matrix Σ = (1/n)X^T X.
- Explained variance: λ_i / Σλ_j for principal component i.

#### Implementation Steps (How)
1. Implement K-means: random init, assign, update, check convergence.
2. Try k-means++ initialization; compare convergence speed.
3. Use elbow method and silhouette scores to select k.
4. Implement PCA: center data, compute covariance, eigen-decomposition.
5. Project high-D data to 2D; visualize clusters.

#### Instrumentation & Evaluation
- Plot inertia vs k (elbow curve).
- Silhouette plots showing cluster cohesion.
- Scree plot (explained variance vs components).
- 2D scatter with cluster colors; annotate centroids.

#### Common Pitfalls & Debugging
- K-means sensitive to initialization: run multiple times, pick best.
- Assuming spherical clusters: K-means fails on elongated/nested structures.
- Choosing k: no universal rule; combine elbow + silhouette + domain knowledge.
- PCA without centering produces incorrect components.
- Interpreting PCs: linear combinations; may lack semantic meaning.

#### FAQ
- Q: When to use K-means vs hierarchical clustering? A: K-means for large datasets, speed; hierarchical for dendrograms and nested structure.
- Q: How much variance to retain in PCA? A: 80-95% common; depends on downstream task sensitivity.
- Q: Can PCA help with overfitting? A: Yes—reduces dimensionality acts as regularization; test empirically.
- Q: What if clusters aren't spherical? A: Try DBSCAN (density-based), Gaussian Mixture Models, or spectral clustering.

#### Extensions & Next Experiments
- DBSCAN for arbitrary-shaped clusters.
- t-SNE/UMAP for non-linear dimensionality reduction (better visualization).
- Gaussian Mixture Models (soft clustering with EM algorithm).
- Autoencoders as non-linear PCA alternative.

#### Alignment to Long-Term Goals
Understanding embeddings and dimensionality reduction prepares you for visualizing LLM hidden states; clustering echoes topic modeling and prompt grouping in fine-tuning analysis; PCA intuition transfers to understanding attention head specialization.

---

### Project 7.8: Testing ML Code and Pipelines

#### Motivation (Why)
Bugs in ML code are subtle—wrong shapes, silent numerical errors, data leakage—and can invalidate entire experiments. Systematic testing catches errors early and builds confidence.

#### Learning Objectives
- Write unit tests for ML components (data generators, metrics, transformations).
- Test data pipelines for correctness and leakage prevention.
- Implement property-based tests for ML invariants.
- Build integration tests for end-to-end workflows.
- Understand what makes ML testing different from standard software testing.

#### Conceptual Core
- ML code has unique failure modes: shape mismatches, numerical instability, stochastic behavior.
- Tests serve as executable documentation and regression prevention.
- Test-driven development (TDD) clarifies interfaces before implementation.
- Property-based testing verifies invariants across input distributions.

#### Mathematical Foundations
- Numerical stability tests: assert abs(computed - expected) < ε with appropriate tolerances.
- Statistical tests: distributional properties (mean, variance) within confidence intervals.
- Invariants: transformations preserve properties (e.g., scaling doesn't change correlations).

#### Implementation Steps (How)
1. **Unit tests for utilities**:
   - Test data generators produce correct shapes and distributions.
   - Test custom metrics against known values.
   - Test preprocessing functions (scaling, encoding).

2. **Integration tests for pipelines**:
   - Test full train/predict workflow end-to-end.
   - Verify no data leakage between train/test splits.
   - Test reproducibility with fixed random seeds.

3. **Property-based tests**:
   - Gradient checks: numerical vs analytical gradients.
   - Inverse operations: encode → decode → original.
   - Invariants: predictions unchanged by irrelevant feature permutations.

4. **Regression tests**:
   - Save baseline outputs; detect unintended changes.
   - Test model serialization/deserialization.

#### Test Examples (Code)
```python
import pytest
import numpy as np
from utils.metrics import mean_squared_error, accuracy
from utils.data_generators import generate_linear_data

def test_mse_known_values():
    """Test MSE against hand-calculated values."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 2.9])
    expected = 0.01  # (0.1^2 + 0.1^2 + 0.1^2) / 3
    assert abs(mean_squared_error(y_true, y_pred) - expected) < 1e-6

def test_mse_perfect_predictions():
    """Perfect predictions should give zero MSE."""
    y = np.random.randn(100)
    assert mean_squared_error(y, y) < 1e-10

def test_data_generator_shape():
    """Check data generator produces correct shapes."""
    X, y = generate_linear_data(n_samples=50, n_features=3)
    assert X.shape == (50, 3)
    assert y.shape == (50,)

def test_data_generator_linear_relationship():
    """Generated data should have strong linear correlation."""
    X, y = generate_linear_data(n_samples=1000, noise=0.1)
    correlation = np.corrcoef(X.ravel(), y)[0, 1]
    assert abs(correlation) > 0.9  # Strong linear relationship

def test_scaler_preserves_shape():
    """Scaling shouldn't change data shape."""
    from sklearn.preprocessing import StandardScaler
    X = np.random.randn(100, 5)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    assert X_scaled.shape == X.shape

def test_train_test_split_no_overlap():
    """Train and test sets should have no common samples."""
    from sklearn.model_selection import train_test_split
    X = np.arange(100).reshape(-1, 1)
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    train_set = set(X_train.ravel())
    test_set = set(X_test.ravel())
    assert len(train_set.intersection(test_set)) == 0

@pytest.mark.parametrize("noise", [0.1, 0.5, 1.0])
def test_model_fits_with_varying_noise(noise):
    """Model should fit data across noise levels."""
    from sklearn.linear_model import LinearRegression
    X, y = generate_linear_data(n_samples=100, noise=noise)
    model = LinearRegression()
    model.fit(X, y)
    score = model.score(X, y)
    assert score > 0.5  # Should explain >50% variance even with noise
```

#### Instrumentation & Evaluation
- Test coverage reports: aim for >80% coverage on utility functions.
- Continuous integration: run tests automatically on commits.
- Test execution time: keep unit tests fast (<1s each).
- Mutation testing (advanced): verify tests actually catch bugs.

#### Common Pitfalls & Debugging
- **Flaky tests**: Random seeds not set; use `np.random.seed()` and `random.seed()`.
- **Overly strict tolerances**: Numerical precision varies; use appropriate ε (1e-6 for float32, 1e-10 for float64).
- **Testing implementation details**: Test behavior/interfaces, not internal mechanics.
- **Ignoring edge cases**: Test empty inputs, single samples, extreme values.
- **Slow tests**: Mock expensive operations (I/O, large computations); use smaller test datasets.

#### FAQ
- Q: How much testing is enough? A: Critical paths (data loading, metric computation, model training loop) should have tests; aim for confidence, not 100% coverage.
- Q: Should I test sklearn/PyTorch functions? A: No—trust established libraries; test *your* code that uses them.
- Q: How to test stochastic algorithms? A: Set random seeds for reproducibility; test statistical properties (mean, variance) with tolerance intervals.
- Q: What about testing neural networks? A: Test shapes, gradient flow (numerical checks), overfitting on tiny datasets (sanity check), and output ranges.
- Q: TDD for ML? A: Useful for utilities and pipelines; less practical for exploratory model development.

#### Extensions & Next Experiments
- Hypothesis library for property-based testing (automatic input generation).
- Pytest fixtures for reusable test data and models.
- Doctests for inline documentation examples.
- Integration with CI/CD (GitHub Actions, pre-commit hooks).
- Test data versioning with DVC (Data Version Control).

#### Alignment to Long-Term Goals
Testing discipline scales to complex systems; when fine-tuning LLMs, you'll test: tokenization correctness, data pipeline integrity, evaluation metric implementation, checkpoint loading/saving, and inference outputs—all preventing costly bugs in long-running experiments.

---

### Project 8: Cross-Validation and Data Splits

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
This mirrors how you’ll run Mistral fine-tuning: clear goals, solid validation, disciplined experimentation, thorough analysis.

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

Total: 8-12 weeks before moving to Mistral/LLM training.

---

## Next Steps

1. Start with Project 1; code from scratch and instrument heavily.
2. Document everything—logs, plots, insights.
3. Don’t rush; depth over breadth.
4. After Project 11, proceed to transformers/pretraining with confidence.

Once you finish Project 11, you'll be ready to return to the Mistral 7B project with solid conceptual foundations.
