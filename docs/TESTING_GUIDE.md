# Testing Machine Learning Code: A Comprehensive Guide

## Why Testing Matters More in ML

Machine learning code has unique failure modes that traditional software doesn't encounter:

| Traditional Software | ML Code |
|---------------------|---------|
| Deterministic outputs | Stochastic behavior |
| Logic errors obvious | Silent numerical errors |
| Type errors caught early | Shape mismatches can propagate |
| Binary pass/fail | Degraded performance may go unnoticed |
| Static inputs | Distribution shifts over time |

**Core Insight**: A bug in ML code might not crash—it might just make your model perform worse, and you won't know why.

---

## Testing Philosophy for ML

### Three Layers of Testing

1. **Unit Tests**: Individual components (data generators, metrics, transformations)
2. **Integration Tests**: End-to-end pipelines (data → training → evaluation)
3. **Property Tests**: Mathematical invariants and statistical properties

### What to Test

✅ **Always Test**:
- Data loading and preprocessing
- Custom metrics and loss functions
- Model architecture (shapes, forward pass)
- Training loop mechanics
- Evaluation pipelines
- Checkpoint save/load

⚠️ **Conditionally Test**:
- Complex feature engineering
- Custom optimizers
- Data augmentation correctness

❌ **Don't Test**:
- Third-party libraries (sklearn, PyTorch, MLX)
- Hyperparameter values (those are experiments, not tests)
- Model performance thresholds (those are validation, not tests)

---

## Practical Testing Patterns

### Pattern 1: Shape Testing

```python
def test_model_output_shape():
    """Most common ML bug: wrong tensor shapes."""
    model = MyModel(input_dim=10, hidden_dim=20, output_dim=5)
    x = torch.randn(32, 10)  # Batch of 32
    
    output = model(x)
    
    assert output.shape == (32, 5), f"Expected (32, 5), got {output.shape}"
```

**Why it matters**: Shape errors propagate silently until they cause NaN or crash much later.

### Pattern 2: Known-Value Testing

```python
def test_mse_with_known_values():
    """Test metric against hand-calculated value."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    
    result = mean_squared_error(y_true, y_pred)
    
    assert result == 0.0, "Perfect predictions should give 0 MSE"

def test_softmax_sums_to_one():
    """Softmax output should be valid probability distribution."""
    logits = torch.randn(10, 5)
    probs = torch.softmax(logits, dim=1)
    
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones(10)), "Softmax must sum to 1"
```

**Why it matters**: Validates implementation correctness with trivial, verifiable cases.

### Pattern 3: Invariant Testing

```python
def test_scaling_preserves_correlation():
    """Standardization shouldn't change correlations."""
    from sklearn.preprocessing import StandardScaler
    
    X = np.random.randn(100, 5)
    corr_before = np.corrcoef(X.T)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    corr_after = np.corrcoef(X_scaled.T)
    
    assert np.allclose(corr_before, corr_after, atol=1e-10)

def test_data_augmentation_preserves_label():
    """Augmented images should keep same label."""
    image, label = dataset[0]
    augmented = augment(image)
    
    # Label shouldn't change after augmentation
    assert get_label(augmented) == label
```

**Why it matters**: Ensures transformations don't violate mathematical properties.

### Pattern 4: Overfitting Sanity Check

```python
def test_model_can_overfit_tiny_dataset():
    """Model should memorize 2 samples (sanity check for capacity)."""
    model = MyNeuralNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Tiny dataset
    X = torch.randn(2, 10)
    y = torch.tensor([0, 1])
    
    # Train for 100 iterations
    for _ in range(100):
        pred = model(X)
        loss = F.cross_entropy(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Should achieve perfect accuracy on 2 samples
    final_pred = model(X).argmax(dim=1)
    assert torch.equal(final_pred, y), "Failed to overfit 2 samples"
```

**Why it matters**: If model can't overfit tiny data, something is fundamentally broken (architecture, loss, optimizer).

### Pattern 5: Gradient Flow Testing

```python
def test_gradients_flow_to_all_parameters():
    """All parameters should receive gradients."""
    model = TransformerModel()
    x = torch.randint(0, 1000, (2, 10))
    y = torch.randint(0, 1000, (2, 10))
    
    output = model(x)
    loss = F.cross_entropy(output.view(-1, 1000), y.view(-1))
    loss.backward()
    
    # Check every parameter has non-zero gradient
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.abs(param.grad).sum() > 0, f"Zero gradient for {name}"
```

**Why it matters**: Catches frozen layers, gradient vanishing, or disconnected computation graphs.

### Pattern 6: Reproducibility Testing

```python
def test_deterministic_with_seed():
    """Same seed should give same results."""
    def train_with_seed(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        model = SimpleModel()
        X = torch.randn(10, 5)
        y = torch.randint(0, 2, (10,))
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        for _ in range(10):
            pred = model(X)
            loss = F.cross_entropy(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        return model.state_dict()
    
    weights1 = train_with_seed(42)
    weights2 = train_with_seed(42)
    
    # Weights should be identical
    for key in weights1:
        assert torch.equal(weights1[key], weights2[key])
```

**Why it matters**: Reproducibility is essential for debugging and scientific validity.

### Pattern 7: Data Leakage Testing

```python
def test_no_train_test_leakage():
    """Train and test sets must be disjoint."""
    X = np.arange(1000)
    y = np.arange(1000)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    train_set = set(X_train)
    test_set = set(X_test)
    
    assert len(train_set.intersection(test_set)) == 0

def test_scaler_fit_only_on_train():
    """Scaler should only see training data during fit."""
    from sklearn.preprocessing import StandardScaler
    
    X_train = np.random.randn(100, 5)
    X_test = np.random.randn(20, 5)
    
    scaler = StandardScaler()
    scaler.fit(X_train)  # Only fit on train
    
    # Statistics should come from train set only
    assert scaler.mean_.shape == (5,)
    assert not np.allclose(scaler.mean_, X_test.mean(axis=0))
```

**Why it matters**: Data leakage is the #1 cause of overly optimistic results.

---

## Testing ML Pipelines

### Integration Test Template

```python
class TestMLPipeline:
    """End-to-end pipeline tests."""
    
    @pytest.fixture
    def sample_data(self):
        """Create reproducible test data."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def test_full_pipeline(self, sample_data):
        """Test complete train → predict workflow."""
        X_train, X_test, y_train, y_test = sample_data
        
        # Pipeline
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression()
        model.fit(X_train_scaled, y_train)
        
        predictions = model.predict(X_test_scaled)
        
        # Assertions
        assert predictions.shape == y_test.shape
        assert set(predictions).issubset({0, 1})  # Only valid classes
        assert accuracy_score(y_test, predictions) > 0.5  # Better than random
    
    def test_pipeline_reproducibility(self, sample_data):
        """Pipeline should give same results with same seed."""
        X_train, X_test, y_train, y_test = sample_data
        
        def run_pipeline():
            np.random.seed(42)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            model = LogisticRegression(random_state=42)
            model.fit(X_scaled, y_train)
            return model.predict(scaler.transform(X_test))
        
        pred1 = run_pipeline()
        pred2 = run_pipeline()
        
        assert np.array_equal(pred1, pred2)
```

---

## Testing Transformers and LLMs

### Transformer-Specific Tests

```python
def test_attention_causality():
    """Causal mask should prevent attending to future."""
    seq_len = 10
    attention = CausalSelfAttention(d_model=64, num_heads=4)
    
    x = torch.randn(1, seq_len, 64)
    output = attention(x)
    
    # Change future tokens - shouldn't affect past predictions
    x_modified = x.clone()
    x_modified[:, 5:, :] = torch.randn(1, 5, 64)
    output_modified = attention(x_modified)
    
    # First 5 positions should be identical
    assert torch.allclose(output[:, :5, :], output_modified[:, :5, :])

def test_positional_encoding_addition():
    """Positional encodings should be added, not concatenated."""
    d_model = 128
    pos_enc = PositionalEncoding(d_model=d_model, max_len=100)
    
    x = torch.randn(2, 50, d_model)
    output = pos_enc(x)
    
    # Shape should be unchanged (added, not concatenated)
    assert output.shape == x.shape

def test_tokenizer_special_tokens():
    """Tokenizer should handle special tokens correctly."""
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    text = "Hello world"
    tokens = tokenizer.encode(text, add_special_tokens=True)
    
    # Should include BOS/EOS
    assert len(tokens) > len(text.split())
    
    # Decode should work
    decoded = tokenizer.decode(tokens)
    assert "Hello" in decoded and "world" in decoded
```

---

## Property-Based Testing with Hypothesis

For ML, property-based testing automatically generates test cases:

```python
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as npst

@given(npst.arrays(dtype=np.float64, shape=st.tuples(
    st.integers(1, 100),  # num samples
    st.integers(1, 20)     # num features
)))
def test_standardization_zero_mean(X):
    """Standardized data should have zero mean."""
    from sklearn.preprocessing import StandardScaler
    
    # Skip constant features
    if np.std(X, axis=0).min() == 0:
        return
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    means = X_scaled.mean(axis=0)
    assert np.allclose(means, 0, atol=1e-10)

@given(
    st.integers(1, 100),  # n_samples
    st.integers(1, 10),   # n_features
    st.floats(0.01, 1.0)  # noise level
)
def test_linear_regression_improves_with_less_noise(n, d, noise):
    """Lower noise should give better R² score."""
    from sklearn.linear_model import LinearRegression
    
    X = np.random.randn(n, d)
    y_true = X.sum(axis=1)
    y_noisy = y_true + noise * np.random.randn(n)
    
    model = LinearRegression()
    model.fit(X, y_noisy)
    score = model.score(X, y_noisy)
    
    # With low noise, should get high R²
    if noise < 0.1:
        assert score > 0.9
```

---

## CI/CD Integration

### pytest Configuration (`pytest.ini`)

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --tb=short
    --strict-markers
    -p no:warnings
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    gpu: marks tests requiring GPU
```

### GitHub Actions Workflow (`.github/workflows/test.yml`)

```yaml
name: Tests

on: [push, pull_request]

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
    
    - name: Run tests
      run: |
        pytest tests/ --cov=utils --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

---

## Test Organization

### Recommended Structure

```
learning-ml-to-llm/
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   ├── data_generators.py
│   └── visualization.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures
│   ├── test_metrics.py
│   ├── test_data_generators.py
│   ├── test_preprocessing.py
│   └── integration/
│       ├── test_pipeline.py
│       └── test_full_training.py
└── pytest.ini
```

### Shared Fixtures (`conftest.py`)

```python
import pytest
import numpy as np
import torch

@pytest.fixture
def sample_regression_data():
    """Standard regression dataset for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X.sum(axis=1) + 0.1 * np.random.randn(100)
    return X, y

@pytest.fixture
def sample_classification_data():
    """Standard classification dataset for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y

@pytest.fixture
def simple_model():
    """Tiny model for fast testing."""
    return torch.nn.Sequential(
        torch.nn.Linear(5, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 2)
    )
```

---

## Testing Checklist

Before committing code, ensure:

- [ ] All data generators have shape tests
- [ ] Custom metrics tested against known values
- [ ] Model architecture tested with forward pass
- [ ] Gradient flow verified on small input
- [ ] Overfitting test passes on tiny dataset
- [ ] No data leakage in train/test splits
- [ ] Preprocessing is tested for correctness
- [ ] Reproducibility verified with fixed seeds
- [ ] Integration test covers full pipeline
- [ ] Tests run in <10 seconds (or marked as slow)

---

## Resources

- **pytest documentation**: https://docs.pytest.org/
- **Hypothesis**: https://hypothesis.readthedocs.io/
- **Testing ML Code (Google)**: Research on testing practices
- **Property-Based Testing for ML**: Blog posts and tutorials

---

## Key Takeaways

1. **Test behavior, not implementation**: Focus on inputs/outputs, not internal mechanics.
2. **Start simple**: Test known values and edge cases before complex scenarios.
3. **Automate everything**: Tests should run automatically on every commit.
4. **Fast feedback**: Keep unit tests fast; move slow tests to integration suite.
5. **Test your code, trust libraries**: Don't test sklearn/PyTorch; test your usage of them.
6. **Catch bugs early**: A 10-second test can save hours of debugging failed experiments.

**Remember**: In ML, silent failures are common. Testing is your early warning system.
