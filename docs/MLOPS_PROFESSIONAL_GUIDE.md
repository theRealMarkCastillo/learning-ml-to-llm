# MLOps & Professional ML Development Guide

## Overview
This guide covers production ML practices essential for professional AI development: experiment tracking, model versioning, A/B testing, monitoring, and deployment strategies.

---

## Part 1: Experiment Tracking & Management

### Why Experiment Tracking Matters

**The Problem**: Without systematic tracking:
- "Which model did we deploy?" → Can't reproduce
- "Why did this perform better?" → Lost hyperparameters
- "Has performance degraded?" → No baseline comparison

**The Solution**: Track everything—code, data, config, metrics, artifacts.

### Tools Overview

| Tool | Best For | Key Features |
|------|----------|--------------|
| **Weights & Biases** | Deep learning, visualization | Real-time dashboards, sweeps, artifacts |
| **MLflow** | General ML, open-source | Model registry, deployment, tracking |
| **TensorBoard** | PyTorch/TF visualization | Scalars, distributions, graphs |
| **Neptune.ai** | Team collaboration | Notebooks, comparison, metadata |

### Implementing Experiment Tracking with W&B

```python
import wandb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Initialize tracking
wandb.init(
    project="ml-learning-experiments",
    config={
        "model": "random_forest",
        "n_estimators": 100,
        "max_depth": 10,
        "dataset": "iris",
        "random_state": 42
    }
)

# Access config
config = wandb.config

# Load data
X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=config.random_state
)

# Train
model = RandomForestClassifier(
    n_estimators=config.n_estimators,
    max_depth=config.max_depth,
    random_state=config.random_state
)
model.fit(X_train, y_train)

# Evaluate and log
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

wandb.log({
    "train_accuracy": train_acc,
    "test_accuracy": test_acc,
    "train_test_gap": train_acc - test_acc
})

# Log model
wandb.sklearn.plot_classifier(model, X_train, X_test, y_train, y_test)
wandb.save('model.pkl')

wandb.finish()
```

### Advanced: Hyperparameter Sweeps

```python
# Define sweep configuration
sweep_config = {
    'method': 'bayes',  # bayes, grid, random
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'n_estimators': {
            'values': [50, 100, 200, 500]
        },
        'max_depth': {
            'distribution': 'int_uniform',
            'min': 5,
            'max': 30
        },
        'min_samples_split': {
            'values': [2, 5, 10]
        }
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project="ml-learning-experiments")

# Define training function
def train():
    with wandb.init():
        config = wandb.config
        
        model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_split=config.min_samples_split
        )
        
        model.fit(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        wandb.log({"test_accuracy": test_acc})

# Run sweep
wandb.agent(sweep_id, train, count=20)
```

### MLflow Alternative

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# Set experiment
mlflow.set_experiment("classical-ml-experiments")

with mlflow.start_run(run_name="rf_baseline"):
    # Log parameters
    n_estimators = 100
    max_depth = 10
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    # Train
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)
    
    # Evaluate and log metrics
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)
    
    # Log model
    mlflow.sklearn.log_model(model, "model")
    
    # Log artifacts
    import matplotlib.pyplot as plt
    plt.figure()
    # ... create plot ...
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")

# Later: Load model
logged_model = 'runs:/<RUN_ID>/model'
loaded_model = mlflow.sklearn.load_model(logged_model)
```

---

## Part 2: Model Versioning & Registry

### Why Version Models?

- **Reproducibility**: Recreate any deployed model
- **Rollback**: Revert to previous version if issues arise
- **Comparison**: Track performance across versions
- **Audit**: Know what's running where

### Model Registry with MLflow

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model from run
run_id = "abc123..."
model_uri = f"runs:/{run_id}/model"

mlflow.register_model(
    model_uri=model_uri,
    name="iris_classifier"
)

# Transition to production
client.transition_model_version_stage(
    name="iris_classifier",
    version=3,
    stage="Production"
)

# Load production model
model = mlflow.pyfunc.load_model(
    model_uri="models:/iris_classifier/Production"
)

# Add description and tags
client.update_model_version(
    name="iris_classifier",
    version=3,
    description="RF with optimized hyperparameters. Test acc: 0.96"
)

client.set_model_version_tag(
    name="iris_classifier",
    version=3,
    key="validation_status",
    value="passed"
)
```

### Model Card Template

```markdown
# Model Card: Iris Classifier v3

## Model Details
- **Model Type**: Random Forest Classifier
- **Version**: 3.0
- **Date**: 2025-11-08
- **Developer**: Your Team
- **License**: MIT

## Intended Use
- **Primary Use**: Classify iris species from flower measurements
- **Primary Users**: Botanists, ML learners
- **Out-of-Scope**: Other flower species, damaged specimens

## Training Data
- **Dataset**: Iris dataset (150 samples, 4 features, 3 classes)
- **Preprocessing**: StandardScaler fitted on training set
- **Splits**: 80% train, 20% test

## Performance
- **Train Accuracy**: 0.98
- **Test Accuracy**: 0.96
- **Per-class F1**: Setosa=1.0, Versicolor=0.95, Virginica=0.94

## Limitations
- Small dataset; may not generalize to wild iris varieties
- Sensitive to feature scaling
- Assumes complete feature availability

## Ethical Considerations
- Low risk application
- No sensitive attributes
```

---

## Part 3: A/B Testing & Online Experimentation

### Statistical Foundations

#### Sample Size Calculation

```python
import numpy as np
from scipy import stats

def calculate_sample_size(baseline_rate, mde, alpha=0.05, power=0.8):
    """
    Calculate required sample size for A/B test.
    
    Args:
        baseline_rate: Current conversion/success rate
        mde: Minimum detectable effect (e.g., 0.02 for 2% absolute lift)
        alpha: Significance level (Type I error rate)
        power: Statistical power (1 - Type II error rate)
    
    Returns:
        Required sample size per variant
    """
    # Two-sided z-test
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    p1 = baseline_rate
    p2 = baseline_rate + mde
    p_pooled = (p1 + p2) / 2
    
    n = (2 * p_pooled * (1 - p_pooled) * (z_alpha + z_beta)**2) / (mde**2)
    
    return int(np.ceil(n))

# Example: detect 2% absolute improvement from 10% baseline
n_required = calculate_sample_size(
    baseline_rate=0.10,
    mde=0.02,
    alpha=0.05,
    power=0.8
)
print(f"Required sample size per variant: {n_required}")
```

#### Statistical Significance Testing

```python
def ab_test_analysis(control_conversions, control_total, 
                     treatment_conversions, treatment_total,
                     alpha=0.05):
    """
    Analyze A/B test results with confidence intervals.
    """
    # Conversion rates
    p_control = control_conversions / control_total
    p_treatment = treatment_conversions / treatment_total
    
    # Pooled proportion
    p_pooled = (control_conversions + treatment_conversions) / (control_total + treatment_total)
    
    # Standard error
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/treatment_total))
    
    # Z-statistic
    z = (p_treatment - p_control) / se
    
    # P-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    # Confidence interval for difference
    z_crit = stats.norm.ppf(1 - alpha/2)
    ci_lower = (p_treatment - p_control) - z_crit * se
    ci_upper = (p_treatment - p_control) + z_crit * se
    
    # Relative lift
    relative_lift = (p_treatment - p_control) / p_control if p_control > 0 else np.nan
    
    results = {
        'control_rate': p_control,
        'treatment_rate': p_treatment,
        'absolute_lift': p_treatment - p_control,
        'relative_lift': relative_lift,
        'z_statistic': z,
        'p_value': p_value,
        'significant': p_value < alpha,
        'ci_95': (ci_lower, ci_upper)
    }
    
    return results

# Example
results = ab_test_analysis(
    control_conversions=100,
    control_total=1000,
    treatment_conversions=130,
    treatment_total=1000,
    alpha=0.05
)

print(f"Control rate: {results['control_rate']:.2%}")
print(f"Treatment rate: {results['treatment_rate']:.2%}")
print(f"Absolute lift: {results['absolute_lift']:.2%}")
print(f"Relative lift: {results['relative_lift']:.2%}")
print(f"P-value: {results['p_value']:.4f}")
print(f"Significant: {results['significant']}")
print(f"95% CI: [{results['ci_95'][0]:.2%}, {results['ci_95'][1]:.2%}]")
```

### Multi-Armed Bandits (Alternative to A/B)

```python
class EpsilonGreedyBandit:
    """
    Epsilon-greedy multi-armed bandit for online optimization.
    Balances exploration and exploitation.
    """
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
    
    def select_arm(self):
        """Select arm using epsilon-greedy strategy."""
        if np.random.random() < self.epsilon:
            # Explore: random arm
            return np.random.randint(self.n_arms)
        else:
            # Exploit: best arm
            return np.argmax(self.values)
    
    def update(self, arm, reward):
        """Update estimates after observing reward."""
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        # Incremental average
        self.values[arm] = value + (reward - value) / n
    
    def get_best_arm(self):
        """Return arm with highest estimated value."""
        return np.argmax(self.values)

# Simulation
bandit = EpsilonGreedyBandit(n_arms=3, epsilon=0.1)
true_rewards = [0.1, 0.15, 0.12]  # Unknown to algorithm

for t in range(1000):
    arm = bandit.select_arm()
    reward = np.random.binomial(1, true_rewards[arm])
    bandit.update(arm, reward)

print(f"True best arm: {np.argmax(true_rewards)}")
print(f"Learned best arm: {bandit.get_best_arm()}")
print(f"Estimated values: {bandit.values}")
print(f"Arm pulls: {bandit.counts}")
```

### Thompson Sampling (Bayesian Alternative)

```python
class ThompsonSamplingBandit:
    """
    Thompson Sampling for binary rewards (Beta-Bernoulli).
    More sophisticated than epsilon-greedy.
    """
    def __init__(self, n_arms):
        self.n_arms = n_arms
        # Beta(1, 1) = uniform prior
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)
    
    def select_arm(self):
        """Sample from posterior and select best."""
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)
    
    def update(self, arm, reward):
        """Update posterior after observing reward."""
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward
    
    def get_expected_values(self):
        """Return expected value for each arm."""
        return self.alpha / (self.alpha + self.beta)

# Simulation
ts_bandit = ThompsonSamplingBandit(n_arms=3)
true_rewards = [0.1, 0.15, 0.12]

for t in range(1000):
    arm = ts_bandit.select_arm()
    reward = np.random.binomial(1, true_rewards[arm])
    ts_bandit.update(arm, reward)

print(f"Expected values: {ts_bandit.get_expected_values()}")
```

---

## Part 4: Model Monitoring in Production

### Key Metrics to Monitor

1. **Model Performance Metrics**
   - Accuracy, precision, recall, F1
   - Latency (p50, p95, p99)
   - Throughput (requests/second)

2. **Data Quality Metrics**
   - Missing value rates
   - Feature distributions
   - Out-of-range values

3. **Data Drift Metrics**
   - Population Stability Index (PSI)
   - KL divergence
   - Kolmogorov-Smirnov statistic

4. **Concept Drift Metrics**
   - Performance degradation over time
   - Prediction distribution shifts

### Implementing Data Drift Detection

```python
import numpy as np
from scipy import stats

def calculate_psi(expected, actual, buckets=10):
    """
    Calculate Population Stability Index.
    PSI < 0.1: No significant change
    0.1 <= PSI < 0.2: Moderate change
    PSI >= 0.2: Significant change
    """
    # Create bins
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)
    
    # Calculate % in each bucket
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    
    # Avoid division by zero
    expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
    actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
    
    # Calculate PSI
    psi = np.sum((actual_percents - expected_percents) * 
                 np.log(actual_percents / expected_percents))
    
    return psi

# Example
training_data = np.random.normal(0, 1, 10000)
production_data_ok = np.random.normal(0, 1, 1000)  # Same distribution
production_data_drift = np.random.normal(0.5, 1.2, 1000)  # Drifted

psi_ok = calculate_psi(training_data, production_data_ok)
psi_drift = calculate_psi(training_data, production_data_drift)

print(f"PSI (no drift): {psi_ok:.4f}")  # Should be < 0.1
print(f"PSI (with drift): {psi_drift:.4f}")  # Should be > 0.2
```

### KS Test for Distribution Shift

```python
def detect_distribution_shift(reference, current, threshold=0.05):
    """
    Use Kolmogorov-Smirnov test to detect distribution changes.
    """
    statistic, p_value = stats.ks_2samp(reference, current)
    
    result = {
        'statistic': statistic,
        'p_value': p_value,
        'shift_detected': p_value < threshold
    }
    
    return result

# Example
reference_dist = np.random.normal(0, 1, 1000)
current_dist_ok = np.random.normal(0, 1, 500)
current_dist_shifted = np.random.normal(0.3, 1.1, 500)

result_ok = detect_distribution_shift(reference_dist, current_dist_ok)
result_shift = detect_distribution_shift(reference_dist, current_dist_shifted)

print(f"No shift - p-value: {result_ok['p_value']:.4f}, detected: {result_ok['shift_detected']}")
print(f"With shift - p-value: {result_shift['p_value']:.4f}, detected: {result_shift['shift_detected']}")
```

### Monitoring Dashboard Setup (Prometheus + Grafana style)

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import random

# Define metrics
prediction_counter = Counter('model_predictions_total', 'Total predictions made')
prediction_latency = Histogram('model_prediction_latency_seconds', 'Prediction latency')
model_accuracy = Gauge('model_accuracy', 'Current model accuracy')
feature_mean = Gauge('feature_mean', 'Mean of input feature', ['feature_name'])

def make_prediction(model, X):
    """Instrumented prediction function."""
    start_time = time.time()
    
    # Make prediction
    prediction = model.predict(X)
    
    # Record metrics
    latency = time.time() - start_time
    prediction_latency.observe(latency)
    prediction_counter.inc()
    
    # Track feature statistics
    for i, feature_name in enumerate(['feature_0', 'feature_1', 'feature_2']):
        feature_mean.labels(feature_name=feature_name).set(X[:, i].mean())
    
    return prediction

# Start metrics server
start_http_server(8000)  # Metrics available at http://localhost:8000

# Simulate predictions
while True:
    X = np.random.randn(10, 3)
    prediction = make_prediction(model, X)
    
    # Update accuracy periodically
    if random.random() < 0.01:
        current_acc = np.random.uniform(0.85, 0.95)
        model_accuracy.set(current_acc)
    
    time.sleep(0.1)
```

---

## Part 5: CI/CD for ML

### GitHub Actions Workflow for ML

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run unit tests
      run: |
        pytest tests/ --cov=utils --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
  
  data-validation:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Validate training data
      run: |
        python scripts/validate_data.py --data-path data/train.csv
  
  train-model:
    runs-on: ubuntu-latest
    needs: data-validation
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Train model
      run: |
        python train.py --config config/baseline.yaml
    
    - name: Evaluate model
      run: |
        python evaluate.py --model models/latest.pkl
    
    - name: Upload model artifact
      uses: actions/upload-artifact@v2
      with:
        name: trained-model
        path: models/latest.pkl
  
  deploy:
    runs-on: ubuntu-latest
    needs: train-model
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Deploy to staging
      run: |
        # Deploy model to staging environment
        echo "Deploying to staging..."
```

### Model Validation Script

```python
# scripts/validate_model.py
"""
Model validation before deployment.
Run as part of CI/CD pipeline.
"""
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
import yaml

def validate_model(model_path, test_data_path, config_path):
    """
    Validate model meets deployment criteria.
    
    Returns:
        bool: True if model passes all checks
    """
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    thresholds = config['deployment_thresholds']
    
    # Load model and data
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    X_test, y_test = load_test_data(test_data_path)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    # Check thresholds
    checks = {
        'accuracy': accuracy >= thresholds['min_accuracy'],
        'precision': precision >= thresholds['min_precision'],
        'recall': recall >= thresholds['min_recall']
    }
    
    # Print results
    print("Validation Results:")
    print(f"  Accuracy: {accuracy:.4f} (threshold: {thresholds['min_accuracy']})")
    print(f"  Precision: {precision:.4f} (threshold: {thresholds['min_precision']})")
    print(f"  Recall: {recall:.4f} (threshold: {thresholds['min_recall']})")
    
    all_passed = all(checks.values())
    print(f"\nValidation: {'PASSED' if all_passed else 'FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--test-data', required=True)
    parser.add_argument('--config', default='config/validation.yaml')
    args = parser.parse_args()
    
    passed = validate_model(args.model, args.test_data, args.config)
    exit(0 if passed else 1)
```

---

## Part 6: Best Practices Checklist

### Before Training
- [ ] Data quality validated
- [ ] Train/val/test splits defined
- [ ] Baseline model established
- [ ] Success metrics defined
- [ ] Experiment tracking configured

### During Training
- [ ] Metrics logged continuously
- [ ] Checkpoints saved regularly
- [ ] Validation performed each epoch
- [ ] Resource usage monitored
- [ ] Early stopping criteria defined

### Before Deployment
- [ ] Model validated on held-out test set
- [ ] Performance meets threshold
- [ ] Model artifacts versioned
- [ ] Model card documented
- [ ] Rollback plan established

### In Production
- [ ] Monitoring dashboards active
- [ ] Alerts configured
- [ ] A/B test running (if applicable)
- [ ] Data drift detection enabled
- [ ] Latency within SLA

---

## Resources

- **MLOps**:
  - [Google MLOps Whitepaper](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
  - [Awesome MLOps](https://github.com/visenger/awesome-mlops)

- **Experiment Tracking**:
  - [Weights & Biases Docs](https://docs.wandb.ai/)
  - [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

- **A/B Testing**:
  - [Trustworthy Online Controlled Experiments](https://experimentguide.com/) (book)
  - [Evan Miller's A/B Tools](https://www.evanmiller.org/ab-testing/)

- **Monitoring**:
  - [Prometheus Documentation](https://prometheus.io/docs/)
  - [Evidently AI](https://www.evidentlyai.com/) (ML monitoring library)

---

## Key Takeaways

1. **Track Everything**: Code, data, config, metrics, artifacts—reproducibility is non-negotiable.
2. **Automate Validation**: CI/CD catches issues before deployment.
3. **Monitor Continuously**: Production models degrade over time; detect drift early.
4. **Test Statistically**: A/B tests and bandits for evidence-based decisions.
5. **Version Models**: Always know what's deployed and be able to rollback.
6. **Document Thoroughly**: Model cards and runbooks save time during incidents.

**Professional ML is 10% modeling, 90% infrastructure and processes.**
