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
# Navigate to your project directory
cd ~/git/learning-ml-to-llm

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

**Add to your `.zshrc` for easy activation:**
```bash
echo 'alias mlenv="cd ~/git/learning-ml-to-llm && source venv/bin/activate"' >> ~/.zshrc
source ~/.zshrc
```

### Step 2: Install Core Dependencies

**Classical ML Libraries:**
```bash
pip install numpy scipy scikit-learn pandas matplotlib seaborn jupyter ipykernel
```

**Deep Learning Libraries (for later phases):**
```bash
pip install torch torchvision torchaudio
pip install mlx mlx-lm  # Apple MLX for M4 optimization
```

**Development Tools:**
```bash
pip install ipython black flake8 pytest
```

**Verify Installation:**
```bash
python3 -c "import numpy, sklearn, pandas, matplotlib; print('All core libraries installed!')"
```

### Step 3: Project Structure Setup

**Create directory structure:**
```bash
mkdir -p projects/phase1_classical_ml/{project01_linear_regression,project02_logistic_regression,project03_multiclass,project04_regularization}
mkdir -p projects/phase1_classical_ml/{project05_decision_trees,project06_random_forests,project07_classification_metrics}
mkdir -p projects/phase1_classical_ml/{project08_cross_validation,project09_svm,project10_feature_engineering,project11_ml_pipeline}
mkdir -p projects/phase2_transformers/{project12_transformer_architecture,project13_tokenization,project14_pretraining,project15_analysis}
mkdir -p projects/phase3_llm_tuning/{project16_mistral_tuning,project17_comparative_analysis}
mkdir -p data/{raw,processed}
mkdir -p notebooks
mkdir -p utils
```

**Create a requirements.txt:**
```bash
cat > requirements.txt << 'EOF'
# Core Scientific Computing
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0

# Machine Learning
scikit-learn>=1.3.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Jupyter
jupyter>=1.0.0
ipykernel>=6.25.0
ipywidgets>=8.0.0

# Deep Learning
torch>=2.0.0
mlx>=0.0.5
mlx-lm>=0.0.5

# Development Tools
ipython>=8.14.0
black>=23.0.0
flake8>=6.0.0
pytest>=7.4.0

# Additional utilities
tqdm>=4.65.0
pyyaml>=6.0
EOF
```

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

**Create a progress log:**
```bash
touch PROGRESS_LOG.md
```

**Log template:**
```markdown
## [Date] - Project [N]: [Name]

### What I Built:
- 

### Key Concepts Learned:
- 

### Challenges Encountered:
- 

### Insights:
- 

### Next Steps:
- 
```

## Phase 4: Utilities and Helper Functions (Build as You Go)

### Create `utils/visualization.py`
Common plotting functions to reuse:
- Loss curves
- Decision boundaries
- Confusion matrices
- Learning rate effects

### Create `utils/data_generators.py`
Synthetic data generation:
- Linear data
- Classification data
- Complex patterns

### Create `utils/metrics.py`
Custom metric implementations:
- Before using sklearn versions
- To understand deeply

## Phase 5: Resource Management on M4

### Memory Monitoring
```bash
# Check system resources
top -l 1 | grep PhysMem
```

### MLX Optimization Tips (for later phases)
```python
# MLX automatically uses unified memory efficiently
import mlx.core as mx

# Check MLX device
print(mx.default_device())
```

### Training Performance
- Your M4 with 64GB RAM is excellent for all exercises
- Classical ML: seconds to minutes
- Tiny transformer pretraining: 4-12 hours
- Mistral fine-tuning: hours per run

## Phase 6: Checkpoint Strategy

### After Each Project
1. **Save notebooks** with outputs
2. **Export to Python scripts** for reusable code
3. **Document in PROGRESS_LOG.md**
4. **Tag learnings** for future reference
5. **Commit to git** (you're already in a repo)

### Git Workflow
```bash
# Initialize if not already
git init

# Create .gitignore
cat > .gitignore << 'EOF'
venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store
*.pt
*.pth
checkpoints/
EOF

# Regular commits
git add .
git commit -m "Complete Project N: [Name]"
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
# Activate environment and start Jupyter
mlenv
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
| Projects 16-17 | 4-6 weeks | Mistral tuning and analysis |
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

**Your Advantage:** M4 with 64GB RAM means no hardware limitations. Use this to:
- Run experiments freely
- Try different configurations
- Keep multiple notebooks open
- Train larger models in later phases

**Research Connection:** As you learn, think about how concepts relate to AI safety:
- How do models generalize?
- What causes memorization vs learning?
- How does fine-tuning change behavior?
- What are failure modes?

This learning path positions you to do rigorous AI safety research with deep technical understanding.

---

## Next Action

Run this command to get started:
```bash
cd ~/git/learning-ml-to-llm
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install numpy scipy scikit-learn pandas matplotlib seaborn jupyter ipykernel
```

Then create your first project notebook and begin!
