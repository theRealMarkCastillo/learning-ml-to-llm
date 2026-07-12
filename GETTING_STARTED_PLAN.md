# Python ML Learning Plan - Getting Started on macOS

## Overview
This plan helps you start the machine learning journey outlined in your learning paths, progressing from classical ML fundamentals through transformers to LLM fine-tuning.

## Phase 1: Environment Setup (Day 1)

### Step 1: Python Environment Setup

**Check Current Python Installation:**
```bash
python3 --version
which python3
```

**Create Dedicated Virtual Environment:**
```bash
# Navigate to your project directory (use whatever path you cloned to)
cd <your-clone-path>/learning-ml-to-llm

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

**Optional: add an alias to your shell rc:**
```bash
# Adjust the path to wherever you cloned the repo
echo 'alias mlenv="cd <your-clone-path>/learning-ml-to-llm && source venv/bin/activate"' >> ~/.zshrc
source ~/.zshrc
```

### Step 2: Install Dependencies

The repo ships a pinned `requirements.txt` with everything you need.
Run `./scripts/setup_environment.sh` (recommended) or install manually:

```bash
pip install -r requirements.txt
```

For Apple Silicon Phase 3 acceleration, the setup script also installs
`mlx` and `mlx-lm` automatically.

### Step 3: Project Structure

The project directories already exist in the repo — you don't need to
create them. Just `cd` into the one for your current project.

### Step 4: Configure Jupyter

**Set up Jupyter kernel:**
```bash
python3 -m ipykernel install --user --name=ml-learning --display-name="Python (ML Learning)"
```

**Launch Jupyter:**
```bash
jupyter notebook
```

## Phase 2: Start with Project 1 - Linear Regression (Days 2-5)

### Project Structure
```bash
cd projects/phase1_classical_ml/project01_linear_regression
```

### Create Initial Notebook
Create `linear_regression_from_scratch.ipynb` with these sections:

**Section 1: Theoretical Foundation**
- Document what you're learning
- Math behind linear regression
- Gradient descent intuition

**Section 2: Generate Synthetic Data**
```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Visualize
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Synthetic Data for Linear Regression')
plt.show()
```

**Section 3: Implement from Scratch**
- Manual MSE loss function
- Manual gradient computation
- Gradient descent loop
- Parameter updates

**Section 4: Visualization**
- Loss curve over iterations
- Learned line vs data points
- Parameter trajectory

**Section 5: Experiments**
- Different learning rates
- Different initial parameters
- Convergence analysis

**Section 6: Compare with sklearn**
- Verify your implementation
- Understand differences

### Daily Schedule for Project 1

**Day 2 (2-3 hours):**
- Read theoretical material on linear regression
- Set up notebook structure
- Generate and visualize data

**Day 3 (2-3 hours):**
- Implement loss function
- Implement gradient computation
- Test with simple examples

**Day 4 (2-3 hours):**
- Implement gradient descent loop
- Add visualization
- Run first training

**Day 5 (2-3 hours):**
- Experiment with hyperparameters
- Compare with sklearn
- Document learnings

## Phase 3: Establish Learning Rhythm (Weeks 2-12)

### Weekly Pattern

**Monday (2 hours):**
- Start new project
- Read theoretical background
- Set up notebook structure

**Tuesday-Wednesday (2-3 hours each):**
- Core implementation
- From-scratch coding

**Thursday (2-3 hours):**
- Add instrumentation
- Visualization

**Friday (2-3 hours):**
- Experimentation
- Compare with libraries

**Weekend (2-4 hours):**
- Documentation
- Reflection
- Plan next project

### Progress Tracking

There's no required progress log file. Use whatever helps you reflect:
a private journal, a git branch per project, or notes inside the notebooks
themselves.

## Phase 4: Utilities and Helper Functions (already in the repo)

`utils/` ships five ready-to-use modules:

- `utils/visualization.py` — loss curves, decision boundaries, confusion matrices, learning-rate comparisons, parameter trajectories
- `utils/data_generators.py` — linear, polynomial, binary, multiclass, sine-wave synthetic data
- `utils/metrics.py` — regression and classification metrics implemented from scratch
- `utils/device.py` — auto-detects MLX / CUDA / MPS / CPU and gives you a unified `get_device()` / `tensor()` / `ensure_seed()`
- `utils/path_helpers.py` — locates the repo root so notebooks work from any subfolder without hard-coded paths

## Phase 5: Resource Management

### Memory Monitoring
```bash
# macOS / Linux: check available memory
free -h                              # Linux
vm_stat | head                       # macOS
```

### Backend Selection (for later phases)

The repo's `utils.device` module auto-detects the best available
backend (MLX → CUDA → MPS → CPU) and exposes it via `get_device()`.

```python
from utils.device import backend_info, get_device, ensure_seed
print(backend_info())
ensure_seed(42)
```

Override with `LEARNING_ML_BACKEND=cpu` (or `mlx` / `cuda` / `mps`).

### Training Performance (rough)

- Classical ML: seconds to minutes on any laptop
- Tiny transformer pretraining: 4-12 hours on Apple Silicon or CUDA
- Qwen2.5 fine-tuning: hours per run on Apple Silicon (MLX) or CUDA

## Phase 6: Checkpoint Strategy

### After Each Project
1. **Save notebooks** with outputs
2. **Export to Python scripts** for reusable code (only when the project explicitly does so — most notebooks stay as notebooks)
3. **Keep your own notes** on what surprised you or what you'd revisit
4. **Commit to git** (you're already in a repo)

### Git Workflow
The repo ships a comprehensive `.gitignore` that covers venvs, checkpoints,
and notebook artifacts. Just `git add` and `git commit` normally:

```bash
git add projects/
git commit -m "Complete Project N: Name"
```

## Phase 7: Learning Resources

### For Each Project

**Before starting:**
1. Read theoretical material (papers, tutorials)
2. Watch video explanations if helpful
3. Sketch math on paper

**During implementation:**
1. Code from scratch first
2. Refer to library docs for comparison
3. Test incrementally

**After completing:**
1. Compare with reference implementations
2. Read source code of sklearn/PyTorch
3. Document what you learned

### Recommended Resources
- **StatQuest videos** - excellent intuitive explanations
- **3Blue1Brown** - visual math understanding
- **Papers with Code** - implementations and benchmarks
- **Scikit-learn documentation** - great tutorials
- **PyTorch tutorials** - for deep learning phase

## Phase 8: Troubleshooting Common Issues

### Python/Pip Issues
```bash
# If packages conflict
pip install --upgrade --force-reinstall [package-name]

# If virtual environment breaks
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Jupyter Issues
```bash
# Restart kernel if imports fail
# In notebook: Kernel -> Restart

# If kernel not showing
python3 -m ipykernel install --user --name=ml-learning --display-name="Python (ML Learning)"
```

### MLX Issues (for later)
```bash
# Ensure latest version
pip install --upgrade mlx mlx-lm

# Check installation
python3 -c "import mlx.core as mx; print(mx.__version__)"
```

## Quick Start Checklist

- [ ] Python 3 installed and working
- [ ] Virtual environment created
- [ ] Core libraries installed (numpy, sklearn, pandas, matplotlib)
- [ ] Jupyter working
- [ ] Project directories created
- [ ] Git initialized
- [ ] First notebook created for Project 1

## Start Command

Once setup is complete:
```bash
# If you added the alias above
mlenv
# Otherwise, from your clone:
source venv/bin/activate
cd projects/phase1_classical_ml/project01_linear_regression
jupyter notebook
```

## Timeline Overview

| Phase | Duration | Focus |
|-------|----------|-------|
| Setup | 1 day | Environment, tools, structure |
| Project 1 | 4 days | Linear regression from scratch |
| Projects 2-4 | 3 weeks | Classification and regularization |
| Projects 5-8 | 4 weeks | Algorithms and validation |
| Projects 9-11 | 3 weeks | SVM, features, end-to-end |
| **Phase 1 Total** | **10-12 weeks** | **Classical ML mastery** |
| Projects 12-13 | 2 weeks | Transformer architecture |
| Project 14 | 2 weeks | Pretraining (the centerpiece) |
| Project 15 | 1 week | Analysis |
| **Phase 2 Total** | **5 weeks** | **Transformer understanding** |
| Projects 16-17 | 4-6 weeks | Qwen2.5 tuning and analysis |
| **Total Journey** | **19-23 weeks** | **Complete understanding** |

## Daily Commitment

**Minimum:** 1-2 hours/day, 5 days/week = 8-12 weeks per phase
**Recommended:** 2-3 hours/day, 5-6 days/week = 10-12 weeks per phase
**Intensive:** 4+ hours/day = faster completion but don't rush understanding

## Success Metrics

**You're on track if:**
- ✅ You can explain concepts without looking at notes
- ✅ Your code works from scratch (not copy-paste)
- ✅ You can predict what will happen when changing parameters
- ✅ You can debug issues by understanding the math
- ✅ You're documenting insights, not just completing tasks

**Red flags:**
- ❌ Copy-pasting code without understanding
- ❌ Moving to next project with unresolved confusion
- ❌ Skipping visualization/instrumentation
- ❌ Not experimenting with hyperparameters
- ❌ Rushing through to "finish"

## Final Notes

**Key Principle:** Deep understanding over speed. Take time to:
- Implement from scratch
- Visualize everything
- Experiment systematically
- Document thoroughly

**Hardware note:** Apple Silicon (M1+) with MLX is the smoothest path through the curriculum, especially for Phase 3. Linux + CUDA works too. CPU-only works but is slower for Project 14 (pretraining) and Phase 3.

**Research connection:** As you learn, think about how concepts relate to AI safety:
- How do models generalize?
- What causes memorization vs learning?
- How does fine-tuning change behavior?
- What are failure modes?

This learning path positions you to do rigorous AI safety research with deep technical understanding.

---

## Next Action

Run this command to get started:
```bash
cd <your-clone-path>/learning-ml-to-llm
./scripts/setup_environment.sh
source venv/bin/activate
jupyter notebook
```

Then open `projects/phase1_classical_ml/project01_linear_regression/linear_regression_from_scratch.ipynb` and begin.
