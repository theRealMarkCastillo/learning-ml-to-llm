# Ethics & Responsible AI Development Guide

## Overview
Professional AI development requires understanding bias, fairness, explainability, and privacy. This guide covers practical techniques for building responsible AI systems.

---

## Part 1: Bias Detection & Mitigation

### Types of Bias in ML

| Bias Type | Description | Example |
|-----------|-------------|---------|
| **Historical Bias** | Training data reflects past discrimination | Hiring data reflecting gender imbalances |
| **Representation Bias** | Some groups underrepresented in data | Facial recognition trained mostly on lighter skin tones |
| **Measurement Bias** | Proxies used instead of true target | Using zip code as proxy for creditworthiness |
| **Aggregation Bias** | One-size-fits-all model ignores subgroup differences | Medical model trained on majority population |
| **Evaluation Bias** | Test set doesn't reflect deployment population | Benchmark biased toward certain demographics |
| **Deployment Bias** | System used differently than intended | Recommendation system amplifying filter bubbles |

### Detecting Bias: Disparate Impact Analysis

```python
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def calculate_disparate_impact(y_true, y_pred, sensitive_attr):
    """
    Calculate disparate impact ratio.
    
    Rule of thumb: DI < 0.8 suggests discrimination
    (80% rule from US employment discrimination law)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (or scores)
        sensitive_attr: Protected attribute (e.g., gender, race)
    
    Returns:
        dict with disparate impact metrics
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'sensitive': sensitive_attr
    })
    
    # Positive prediction rates by group
    positive_rates = df.groupby('sensitive')['y_pred'].mean()
    
    # Disparate impact: min/max ratio
    di_ratio = positive_rates.min() / positive_rates.max()
    
    results = {
        'positive_rates': positive_rates.to_dict(),
        'disparate_impact_ratio': di_ratio,
        'passes_80_rule': di_ratio >= 0.8
    }
    
    return results

# Example
y_true = np.random.binomial(1, 0.5, 1000)
y_pred = np.random.binomial(1, 0.5, 1000)
gender = np.random.choice(['M', 'F'], 1000)

# Simulate bias: higher positive rate for one group
y_pred[gender == 'M'] = np.random.binomial(1, 0.6, (gender == 'M').sum())
y_pred[gender == 'F'] = np.random.binomial(1, 0.4, (gender == 'F').sum())

di_results = calculate_disparate_impact(y_true, y_pred, gender)
print("Disparate Impact Analysis:")
print(f"  Positive rates: {di_results['positive_rates']}")
print(f"  DI ratio: {di_results['disparate_impact_ratio']:.3f}")
print(f"  Passes 80% rule: {di_results['passes_80_rule']}")
```

### Fairness Metrics

```python
def calculate_fairness_metrics(y_true, y_pred, sensitive_attr):
    """
    Calculate multiple fairness metrics across groups.
    """
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'sensitive': sensitive_attr
    })
    
    results = {}
    
    for group in df['sensitive'].unique():
        group_mask = df['sensitive'] == group
        y_true_group = df.loc[group_mask, 'y_true']
        y_pred_group = df.loc[group_mask, 'y_pred']
        
        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
        
        results[group] = {
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,  # False positive rate
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,  # False negative rate
            'selection_rate': (tp + fp) / len(y_true_group)
        }
    
    # Calculate fairness gaps
    groups = list(results.keys())
    fairness_gaps = {}
    
    for metric in ['accuracy', 'fpr', 'fnr', 'selection_rate']:
        values = [results[g][metric] for g in groups]
        fairness_gaps[f'{metric}_gap'] = max(values) - min(values)
    
    return results, fairness_gaps

# Example
results, gaps = calculate_fairness_metrics(y_true, y_pred, gender)

print("\nFairness Metrics by Group:")
for group, metrics in results.items():
    print(f"\n{group}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")

print("\nFairness Gaps:")
for gap_name, gap_value in gaps.items():
    print(f"  {gap_name}: {gap_value:.3f}")
```

### Mitigation Strategies

#### 1. Pre-processing: Reweighting

```python
from sklearn.utils.class_weight import compute_sample_weight

def reweight_training_data(X, y, sensitive_attr):
    """
    Reweight samples to balance representation across groups.
    """
    # Create combined label: (class, sensitive_attribute)
    combined = [f"{y_val}_{sens_val}" for y_val, sens_val 
                in zip(y, sensitive_attr)]
    
    # Compute weights
    weights = compute_sample_weight('balanced', combined)
    
    return weights

# Usage with sklearn
from sklearn.linear_model import LogisticRegression

weights = reweight_training_data(X_train, y_train, sensitive_train)
model = LogisticRegression()
model.fit(X_train, y_train, sample_weight=weights)
```

#### 2. In-processing: Adversarial Debiasing

```python
import torch
import torch.nn as nn

class FairClassifier(nn.Module):
    """
    Classifier with adversarial debiasing.
    Adversary tries to predict sensitive attribute from hidden layer.
    Main classifier learns to fool adversary while making good predictions.
    """
    def __init__(self, input_dim, hidden_dim, n_sensitive):
        super().__init__()
        
        # Main classifier
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.classifier = nn.Linear(hidden_dim, 1)
        
        # Adversary (predicts sensitive attribute)
        self.adversary = nn.Linear(hidden_dim, n_sensitive)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        predictions = self.classifier(features)
        sensitive_pred = self.adversary(features)
        return predictions, sensitive_pred

# Training loop
def train_fair_model(model, train_loader, n_epochs, lambda_adv=1.0):
    """
    Train with adversarial debiasing.
    
    Args:
        lambda_adv: Weight of adversarial loss (higher = more fairness emphasis)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(n_epochs):
        for X_batch, y_batch, sensitive_batch in train_loader:
            # Forward pass
            pred, sensitive_pred = model(X_batch)
            
            # Main classification loss
            loss_class = criterion(pred.squeeze(), y_batch.float())
            
            # Adversarial loss (want adversary to fail)
            loss_adv = criterion(sensitive_pred, sensitive_batch.float())
            
            # Combined loss
            # Classifier maximizes adversary loss, adversary minimizes it
            loss = loss_class - lambda_adv * loss_adv
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

#### 3. Post-processing: Threshold Optimization

```python
def optimize_thresholds_for_fairness(y_true, y_scores, sensitive_attr, 
                                      fairness_metric='equalized_odds'):
    """
    Find group-specific thresholds that satisfy fairness constraint.
    
    Fairness metrics:
    - 'demographic_parity': Equal selection rates
    - 'equalized_odds': Equal TPR and FPR
    - 'equal_opportunity': Equal TPR
    """
    from sklearn.metrics import roc_curve
    
    groups = np.unique(sensitive_attr)
    thresholds = {}
    
    if fairness_metric == 'demographic_parity':
        # Target: equal selection rates
        target_rate = np.mean(y_true)  # Overall positive rate
        
        for group in groups:
            mask = sensitive_attr == group
            # Find threshold giving target_rate selection rate
            group_scores = y_scores[mask]
            thresholds[group] = np.percentile(group_scores, 
                                              (1 - target_rate) * 100)
    
    elif fairness_metric == 'equal_opportunity':
        # Target: equal TPR (recall)
        target_tpr = 0.8  # Desired TPR
        
        for group in groups:
            mask = (sensitive_attr == group) & (y_true == 1)
            if mask.sum() > 0:
                group_scores = y_scores[mask]
                thresholds[group] = np.percentile(group_scores,
                                                  (1 - target_tpr) * 100)
    
    return thresholds

# Usage
thresholds = optimize_thresholds_for_fairness(
    y_true, y_scores, sensitive_attr, 
    fairness_metric='equal_opportunity'
)

# Apply group-specific thresholds
y_pred_fair = np.zeros_like(y_scores)
for group, threshold in thresholds.items():
    mask = sensitive_attr == group
    y_pred_fair[mask] = (y_scores[mask] >= threshold).astype(int)
```

---

## Part 2: Model Explainability

### SHAP (SHapley Additive exPlanations)

```python
import shap
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Create explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global feature importance
shap.summary_plot(shap_values[1], X_test, plot_type="bar")

# Individual prediction explanation
shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][0],
    X_test.iloc[0]
)

# Dependence plot (feature interaction)
shap.dependence_plot("feature_0", shap_values[1], X_test)
```

### LIME (Local Interpretable Model-agnostic Explanations)

```python
import lime
import lime.lime_tabular

# Create LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns,
    class_names=['Negative', 'Positive'],
    mode='classification'
)

# Explain a single prediction
i = 0  # Instance to explain
exp = explainer.explain_instance(
    X_test.iloc[i].values,
    model.predict_proba,
    num_features=10
)

# Visualize
exp.show_in_notebook()

# Get explanation as dict
exp.as_list()
```

### Partial Dependence Plots

```python
from sklearn.inspection import PartialDependenceDisplay

# Show how model output changes with feature values
features = [0, 1, (0, 1)]  # Single features and interaction
PartialDependenceDisplay.from_estimator(
    model, 
    X_train, 
    features,
    target=1
)
plt.tight_layout()
plt.show()
```

### Custom Explainability: Feature Attribution

```python
def permutation_importance(model, X, y, n_repeats=10):
    """
    Calculate feature importance by permutation.
    Measures performance drop when feature is shuffled.
    """
    from sklearn.metrics import accuracy_score
    
    baseline_score = accuracy_score(y, model.predict(X))
    importances = {}
    
    for col in X.columns:
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[col] = np.random.permutation(X_permuted[col])
            score = accuracy_score(y, model.predict(X_permuted))
            scores.append(baseline_score - score)
        
        importances[col] = {
            'mean': np.mean(scores),
            'std': np.std(scores)
        }
    
    # Sort by importance
    importances = dict(sorted(importances.items(), 
                            key=lambda x: x[1]['mean'], 
                            reverse=True))
    
    return importances

# Usage
importances = permutation_importance(model, X_test, y_test)
for feature, scores in importances.items():
    print(f"{feature}: {scores['mean']:.4f} ± {scores['std']:.4f}")
```

---

## Part 3: Privacy-Preserving ML

### Differential Privacy

```python
def add_laplace_noise(data, epsilon, sensitivity):
    """
    Add Laplacian noise for differential privacy.
    
    Args:
        data: Data to protect
        epsilon: Privacy budget (smaller = more private)
        sensitivity: Max change from one record
    
    Returns:
        Noisy data satisfying epsilon-differential privacy
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, data.shape)
    return data + noise

# Example: Private mean
true_mean = X_train.mean()
private_mean = add_laplace_noise(
    true_mean,
    epsilon=0.1,  # Strong privacy
    sensitivity=1.0  # Assuming normalized data
)

print(f"True mean: {true_mean}")
print(f"Private mean: {private_mean}")
```

### DP-SGD (Differentially Private Stochastic Gradient Descent)

```python
def dp_sgd_step(model, X_batch, y_batch, lr, noise_multiplier, max_grad_norm):
    """
    One step of DP-SGD.
    
    Args:
        noise_multiplier: Noise scale relative to clipping norm
        max_grad_norm: Gradient clipping threshold
    """
    # Compute per-example gradients
    loss = compute_loss(model, X_batch, y_batch)
    per_example_grads = compute_per_example_gradients(loss)
    
    # Clip gradients
    clipped_grads = []
    for grad in per_example_grads:
        norm = torch.norm(grad)
        clipped_grad = grad * min(1, max_grad_norm / norm)
        clipped_grads.append(clipped_grad)
    
    # Average clipped gradients
    avg_grad = torch.mean(torch.stack(clipped_grads), dim=0)
    
    # Add noise
    noise_scale = noise_multiplier * max_grad_norm
    noise = torch.randn_like(avg_grad) * noise_scale / len(X_batch)
    private_grad = avg_grad + noise
    
    # Update parameters
    with torch.no_grad():
        for param, grad in zip(model.parameters(), [private_grad]):
            param -= lr * grad
    
    return model
```

### Federated Learning Basics

```python
class FederatedModel:
    """
    Simple federated learning framework.
    Train locally, aggregate globally without sharing data.
    """
    def __init__(self, model_fn):
        self.global_model = model_fn()
        self.client_models = []
    
    def distribute_model(self, n_clients):
        """Send global model to clients."""
        self.client_models = [
            copy.deepcopy(self.global_model) 
            for _ in range(n_clients)
        ]
    
    def train_clients(self, client_datasets, n_epochs):
        """Each client trains on local data."""
        for i, (model, data) in enumerate(zip(self.client_models, client_datasets)):
            X, y = data
            # Local training
            for epoch in range(n_epochs):
                model.fit(X, y)
    
    def aggregate_models(self):
        """Federated averaging."""
        global_params = {}
        
        # Average parameters across clients
        for param_name in self.global_model.get_params():
            param_values = [
                client.get_params()[param_name]
                for client in self.client_models
            ]
            global_params[param_name] = np.mean(param_values, axis=0)
        
        self.global_model.set_params(global_params)
    
    def federated_round(self, client_datasets, n_epochs):
        """One round of federated learning."""
        self.distribute_model(len(client_datasets))
        self.train_clients(client_datasets, n_epochs)
        self.aggregate_models()
```

---

## Part 4: Responsible AI Checklist

### Pre-Development
- [ ] Stakeholder analysis completed
- [ ] Potential harms identified
- [ ] Success metrics include fairness criteria
- [ ] Data collection ethically sourced
- [ ] Privacy requirements defined

### During Development
- [ ] Training data analyzed for bias
- [ ] Multiple fairness metrics evaluated
- [ ] Model explainability implemented
- [ ] Subgroup performance analyzed
- [ ] Edge cases and failure modes documented

### Pre-Deployment
- [ ] Bias audit completed
- [ ] Fairness thresholds met
- [ ] Explanations validated with domain experts
- [ ] Privacy guarantees verified
- [ ] Impact assessment documented
- [ ] Rollback plan established

### Post-Deployment
- [ ] Fairness monitoring active
- [ ] Explanation quality tracked
- [ ] User feedback mechanism implemented
- [ ] Regular bias audits scheduled
- [ ] Incident response plan ready

---

## Part 5: Practical Example: Fair Lending Model

```python
"""
Complete example: Build fair credit scoring model
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# Load data
df = pd.read_csv('credit_data.csv')

# Identify sensitive attribute
sensitive_attr = 'race'
X = df.drop(['default', sensitive_attr], axis=1)
y = df['default']
sensitive = df[sensitive_attr]

# Split data
X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X, y, sensitive, test_size=0.2, stratify=y, random_state=42
)

# 1. Train baseline model
print("1. Baseline Model")
baseline_model = GradientBoostingClassifier()
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)

# 2. Evaluate fairness
print("\n2. Fairness Evaluation")
baseline_fairness, baseline_gaps = calculate_fairness_metrics(
    y_test, y_pred_baseline, s_test
)
print("Baseline fairness gaps:", baseline_gaps)

# 3. Check disparate impact
print("\n3. Disparate Impact")
di_results = calculate_disparate_impact(y_test, y_pred_baseline, s_test)
print(f"DI ratio: {di_results['disparate_impact_ratio']:.3f}")
print(f"Passes 80% rule: {di_results['passes_80_rule']}")

# 4. Apply fairness mitigation (reweighting)
print("\n4. Fair Model Training")
weights = reweight_training_data(X_train, y_train, s_train)
fair_model = GradientBoostingClassifier()
fair_model.fit(X_train, y_train, sample_weight=weights)
y_pred_fair = fair_model.predict(X_test)

# 5. Evaluate fair model
print("\n5. Fair Model Evaluation")
fair_fairness, fair_gaps = calculate_fairness_metrics(
    y_test, y_pred_fair, s_test
)
print("Fair model gaps:", fair_gaps)

di_results_fair = calculate_disparate_impact(y_test, y_pred_fair, s_test)
print(f"Fair DI ratio: {di_results_fair['disparate_impact_ratio']:.3f}")

# 6. Compare overall performance
print("\n6. Performance Comparison")
print(f"Baseline AUC: {roc_auc_score(y_test, baseline_model.predict_proba(X_test)[:, 1]):.3f}")
print(f"Fair model AUC: {roc_auc_score(y_test, fair_model.predict_proba(X_test)[:, 1]):.3f}")

# 7. Generate explanations
print("\n7. Model Explainability")
explainer = shap.TreeExplainer(fair_model)
shap_values = explainer.shap_values(X_test.iloc[:100])
shap.summary_plot(shap_values[1], X_test.iloc[:100], plot_type="bar")

# 8. Document in model card
model_card = {
    'model_type': 'GradientBoostingClassifier',
    'training_date': '2025-11-08',
    'intended_use': 'Credit default prediction',
    'fairness_metrics': fair_gaps,
    'disparate_impact': di_results_fair,
    'performance': {
        'auc': roc_auc_score(y_test, fair_model.predict_proba(X_test)[:, 1])
    },
    'known_limitations': [
        'May underperform on recent immigrants with thin credit files',
        'Temporal drift expected as economic conditions change'
    ]
}

print("\n8. Model Card:", model_card)
```

---

## Resources

- **Fairness**:
  - [Fairness and Machine Learning](https://fairmlbook.org/) (book by Barocas, Hardt, Narayanan)
  - [AI Fairness 360](https://aif360.mybluemix.net/) (IBM toolkit)
  - [Fairlearn](https://fairlearn.org/) (Microsoft toolkit)

- **Explainability**:
  - [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/) (book by Christoph Molnar)
  - [SHAP Documentation](https://shap.readthedocs.io/)
  - [LIME](https://github.com/marcotcr/lime)

- **Privacy**:
  - [Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf) (book by Dwork & Roth)
  - [Opacus](https://opacus.ai/) (PyTorch differential privacy)
  - [TensorFlow Privacy](https://github.com/tensorflow/privacy)

- **General**:
  - [Google's Responsible AI Practices](https://ai.google/responsibility/responsible-ai-practices/)
  - [Microsoft's Responsible AI Resources](https://www.microsoft.com/en-us/ai/responsible-ai-resources)

---

## Key Takeaways

1. **Bias is Inevitable**: All data reflects historical biases; acknowledge and measure them.
2. **Multiple Metrics**: No single fairness metric is perfect; evaluate multiple.
3. **Trade-offs Exist**: Fairness may reduce overall accuracy; document trade-offs.
4. **Explainability is Essential**: Stakeholders need to understand model decisions.
5. **Privacy Matters**: Implement privacy protections, especially for sensitive data.
6. **Continuous Monitoring**: Fairness and bias must be tracked in production.
7. **Document Everything**: Model cards and impact assessments are professional requirements.

**Responsible AI is not optional—it's a core engineering competency.**
