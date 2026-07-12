# 🚀 ML to LLM Learning Journey

**A comprehensive hands-on learning path from classical machine learning through transformers to LLM fine-tuning.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter)](https://jupyter.org/try)

## 🎯 What Is This?

This repository contains a complete, structured learning curriculum that takes you from zero to hero in machine learning and LLMs. Instead of treating large language models as mysterious black boxes, you'll build **genuine understanding from first principles**.

### Who Is This For?

✅ **You should use this if you want to**:
- Understand how ML/LLMs actually work (not just use them)
- Build models from scratch before using libraries
- Prepare for AI safety research or ML engineering roles
- Learn through hands-on projects, not just theory
- Have a systematic path from basics to advanced topics

❌ **This might not be for you if**:
- You just want to use pre-built models (use Hugging Face instead)
- You're looking for a quick weekend tutorial
- You prefer video courses over hands-on coding
- You don't have 4-6 months for deep learning

### What Makes This Different?

**Most ML courses either**:
1. 🚫 Treat models as black boxes (just call APIs)
2. 🚫 Jump straight to deep learning (missing foundations)
3. 🚫 Focus on theory without implementation
4. 🚫 Use frameworks without understanding internals

**This curriculum**:
1. ✅ Implements everything from scratch first
2. ✅ Builds foundations before advanced topics
3. ✅ Balances theory with extensive coding
4. ✅ Teaches *why* before showing library shortcuts

## 🎓 What You'll Learn

By completing this journey, you'll deeply understand:

### Technical Mastery
- 🔢 **Gradient Descent**: How optimization really works
- 📊 **Loss Functions**: MSE, cross-entropy, and why they matter
- 🌳 **Classical ML**: Trees, SVMs, ensembles from scratch
- 🤖 **Transformers**: Self-attention, positional encoding, architecture
- 🔥 **Pretraining**: What happens when models learn language
- 🎯 **Fine-tuning**: LoRA and parameter-efficient methods
- 📈 **Evaluation**: Proper metrics and experimental design

### Practical Skills
- Write ML algorithms from scratch (NumPy only)
- Build and train transformer models (PyTorch)
- Fine-tune production LLMs (MLX on Apple Silicon)
- Design rigorous experiments
- Debug models by understanding internals

### Research Readiness
- Systematic analysis methodology
- Hypothesis-driven experimentation
- Rigorous documentation practices
- Foundation for AI safety research

## Learning Path

### Phase 1: Classical ML Foundation (Weeks 1-12)
**Goal**: Master fundamental ML concepts before approaching deep learning

**Projects 1-11**: Core foundations
- Linear & Logistic Regression from scratch
- Multi-class classification with softmax
- Regularization and overfitting
- Decision trees and random forests
- Classification metrics deep dive
- Cross-validation strategies
- Support Vector Machines
- Feature engineering
- End-to-end ML pipeline

**Bridge Projects** (prepare for transformers):
- 11.5: Neural Networks from scratch (backprop, depth vs width)
- 11.75: RNNs from scratch (BPTT, vanishing gradients, why transformers are better)

**Key Learning**: Gradient descent, loss functions, generalization, proper evaluation, deep learning intuition, sequence modeling

[→ Phase 1 Details](projects/phase1_classical_ml/README.md)

### Phase 2: Transformers & Pretraining (Weeks 13-18)
**Goal**: Build and pretrain a transformer to understand base models

**Bridge Projects** (build intuition before assembly):
- 12.1: Attention Mechanisms from scratch
- 12.25: Embeddings & representation learning via skip-gram

**Core Projects**:
- Build transformer architecture from scratch
- Tokenization and text preprocessing
- **Pretrain tiny transformer on Shakespeare** (4-12 hours on Apple Silicon, similar on CUDA, slower on CPU)
- Analyze pretrained vs random models

**Key Learning**: Self-attention, multi-head attention, embeddings, pretraining dynamics, why base models work

[→ Phase 2 Details](projects/phase2_transformers/README.md)

### Phase 3: LLM Fine-tuning (Weeks 19-22)
**Goal**: Fine-tune Qwen2.5-1.5B-Instruct and analyze behavior changes

**Projects**:
- Instruction tune Qwen2.5-1.5B-Instruct with LoRA (using MLX)
- Comparative analysis: base vs tuned model
- Systematic evaluation and documentation

**Key Learning**: LoRA efficiency, instruction tuning, model evaluation

[→ Phase 3 Details](projects/phase3_llm_tuning/README.md)

## ⚡ Quick Start

### No Hard-Coded Paths
All notebooks now resolve the repository root dynamically instead of using a user-specific absolute path like `/Users/mark/git/learning-ml-to-llm`. Use either the inline helper pattern:

```python
import sys, pathlib
def add_repo_root(markers=("requirements.txt","README.md",".git")):
    here = pathlib.Path.cwd().resolve()
    for candidate in [here] + list(here.parents):
        if any((candidate / m).exists() for m in markers):
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            break
add_repo_root()
```

Or reuse the utility:

```python
from utils.path_helpers import add_repo_root_to_sys_path
add_repo_root_to_sys_path()
```

After this, relative imports like `from utils import metrics` work from any project subfolder without editing paths.

### Automatic Device Selection (GPU / MPS / MLX / CPU)
The repository now includes unified backend auto-detection via `utils.device`.

Priority order:
1. MLX (Apple Silicon) if available (`import mlx.core as mx`).
2. PyTorch CUDA if `torch.cuda.is_available()`.
3. PyTorch MPS if `torch.backends.mps.is_available()`.
4. CPU fallback (torch CPU or pure Python).

Usage in notebooks (already inserted in Phase 2 & 3 transformer notebooks):
```python
from utils.device import get_device, backend_info, tensor, ensure_seed
print("Using backend:", backend_info())
ensure_seed(42)

# Create a tensor on the active backend
x = tensor([[1.0, 2.0], [3.0, 4.0]])
```

Override backend manually:
```bash
export LEARNING_ML_BACKEND=cpu   # options: mlx | cuda | mps | cpu
python scripts/verify_device.py
```

Quick verification script:
```bash
python scripts/verify_device.py
```
This prints the chosen backend and runs a tiny matmul to confirm functionality.

Why this matters:
- Seamless cross-platform execution (Apple Silicon MLX, Linux CUDA, macOS MPS).
- Single import path for device logic keeps notebooks clean.
- Consistent seeding across random, NumPy, torch, and MLX for reproducibility.

See `utils/device.py` for details and helper functions (`backend_name`, `move_to`).

### Prerequisites

- **Python 3.8+** installed
- **4-8GB RAM** minimum (16-32GB recommended for Phase 3)
- **Jupyter** for running notebooks
- **Time commitment**: 10-20 hours/week for 4-6 months
- **Math background**: Basic calculus and linear algebra helpful but not required

### 1. Clone and Setup

```bash
# Clone this repository
git clone https://github.com/<owner>/learning-ml-to-llm.git
cd learning-ml-to-llm

# Automated setup (recommended)
./scripts/setup_environment.sh
```

<details>
<summary>Or manual setup (click to expand)</summary>

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup Jupyter kernel
python3 -m ipykernel install --user --name=ml-learning --display-name="Python (ML Learning)"
```
</details>

### 2. Start Project 1

```bash
# Activate environment
source venv/bin/activate

# Launch Jupyter
jupyter notebook

# Open: projects/phase1_classical_ml/project01_linear_regression/linear_regression_from_scratch.ipynb
```

### 3. Follow the Path

Work through projects sequentially:
1. Complete the notebook
2. Run all experiments
3. Document learnings in your own notes
4. Move to next project

**🎉 You're ready to start learning!**

## Repository Structure

```
learning-ml-to-llm/
├── projects/
│   ├── phase1_classical_ml/       # Projects 1-11 + bridges 11.5, 11.75
│   ├── phase2_transformers/       # Bridges 12.1, 12.25 + projects 12-15
│   └── phase3_llm_tuning/         # Projects 16-17
├── docs/
│   ├── GLOSSARY.md                # ML terminology reference
│   └── LEARNING_OVERVIEW.md       # Learning strategy guide
├── utils/
│   ├── path_helpers.py            # Repo-root discovery (no hard-coded paths)
│   ├── device.py                  # Backend auto-detect (MLX / CUDA / MPS / CPU)
│   ├── visualization.py           # Plotting utilities
│   ├── data_generators.py         # Synthetic data generation
│   └── metrics.py                 # Evaluation metrics
├── data/
│   ├── raw/                       # Raw datasets
│   └── processed/                 # Processed data
├── scripts/
│   ├── setup_environment.sh       # Setup script
│   ├── download_shakespeare.py    # Download training data
│   ├── verify_device.py           # Verify backend selection
│   └── verify_repo.sh             # Sanity-check repo structure
├── tests/                         # pytest suite for utils + extracted modules
├── requirements.txt               # Python dependencies
├── classical_ml_learning_path.md         # Phase 1 deep-dive
├── complete_ml_learning_path_with_pretraining.md  # Phases 1-2 deep-dive
├── qwen_mlx_learning_project.md   # Phase 3 deep-dive
├── GETTING_STARTED_PLAN.md        # Day-1 setup walkthrough
├── QUICK_REFERENCE.md             # Cheat sheet
├── CONTRIBUTING.md                # Repo conventions
└── README.md                      # This file
```

## 💻 Hardware Requirements

### Minimum Specs (Phases 1-2)
- **CPU**: Any modern processor
- **RAM**: 4-8GB
- **Storage**: 5GB
- **OS**: macOS, Linux, or Windows (CPU); Linux/macOS for Phase 2+ acceleration

### Recommended Specs (All Phases)
- **CPU**: Apple Silicon (M1/M2/M3/M4) or modern x86 with CUDA GPU
- **RAM**: 16GB+ (Apple Silicon unified memory is fine)
- **Storage**: 20GB
- **Accelerator**: MLX on Apple Silicon, or CUDA GPU on Linux (8GB+ VRAM)

### What Runs Where

| Phase | Project | Time | RAM Needed |
|-------|---------|------|------------|
| Phase 1 | Classical ML (1-11) | Seconds-Minutes | 2-4GB |
| Phase 2 | Build Transformer (12-13) | Minutes | 1-2GB |
| Phase 2 | Pretrain Tiny Model (14) | 4-12 hours | 3-8GB |
| Phase 3 | Fine-tune Qwen2.5 (16) | Hours | 4-8GB |

**Good news**: Phases 1-2 run on any laptop. Phase 3 wants a real accelerator but the model is small enough that 8GB is enough.

## Learning Approach

### Core Principles
1. **Implement from scratch first** - Understand before using libraries
2. **Visualize everything** - Loss curves, decision boundaries, attention
3. **Experiment systematically** - Vary hyperparameters, observe effects
4. **Document deeply** - Record insights, not just results
5. **Don't rush** - Deep understanding > speed

### Daily Commitment
- **Minimum**: 1-2 hours/day, 5 days/week
- **Recommended**: 2-3 hours/day, 5-6 days/week
- **Total timeline**: 18-22 weeks (~4-5 months)

## Progress Tracking

Keep your own notes as you go — a private learning journal, a git branch per project, or whatever fits your style. There is no required progress log file; do whatever helps you reflect.

## Resources

### Included Documents

**Learning Paths** (pick one and follow it end-to-end):
- `classical_ml_learning_path.md` — Detailed Phase 1 guide (Projects 1-11)
- `complete_ml_learning_path_with_pretraining.md` — Phases 1-2 deep-dive
- `qwen_mlx_learning_project.md` — Phase 3 deep-dive
- `GETTING_STARTED_PLAN.md` — Day-1 setup walkthrough
- `QUICK_REFERENCE.md` — Cheat sheet for the whole path

**Reference**:
- `docs/GLOSSARY.md` — A→Z terminology reference (heavily cross-linked from notebooks)

### External Resources (Recommended)
- **StatQuest** - Intuitive ML explanations
- **3Blue1Brown** - Visual understanding of math
- **Papers with Code** - Implementation references
- **MLX Documentation** - Apple silicon optimization

## Learning Outcomes

By completing this journey, you'll understand:

**Technical Understanding**:
- How gradient descent works at a deep level
- What transformers do and why they work
- How pretraining creates language understanding
- Why fine-tuning is efficient and effective

**Practical Skills**:
- Implement ML/DL algorithms from scratch
- Design and run rigorous experiments
- Evaluate models systematically
- Optimize for Apple silicon (MLX)

**Research Capacity**:
- Analyze model behavior methodically
- Design experiments to test hypotheses
- Document findings rigorously
- Connect to AI safety research

## 🤔 Why This Path?

### The Problem with Traditional ML Education

**Most courses do this**:
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
# ✨ Magic happens ✨
```
You learn to use tools but don't understand what's inside.

**This curriculum does this**:
```python
# Week 1: Build gradient descent from scratch
def gradient_descent(X, y, learning_rate):
    # You write every line
    ...

# Week 13: Build attention mechanism
def self_attention(Q, K, V):
    # You understand every operation
    ...

# Week 15: Watch your model learn language
# You see loss decrease, watch text generation improve

# Week 18: Now use the tools with full understanding
from mlx_lm import load
# You know exactly what this does internally
```

### The Learning Philosophy

1. **Foundations First**: Master optimization on simple problems
2. **Build, Don't Import**: Implement before using libraries  
3. **Visualize Everything**: See what's happening inside
4. **Experiment Systematically**: Change parameters, observe effects
5. **Understand, Then Apply**: Theory + practice = mastery

### Why This Matters

**For ML Engineering**:
- Debug models by understanding internals
- Choose right architectures with confidence
- Optimize training efficiently
- Read papers and implement them

**For AI Safety Research**:
- Understand how fine-tuning changes behavior
- Analyze model responses rigorously
- Design evaluation methodologies
- Document epistemic properties

**For Deep Understanding**:
- Know *why* things work, not just *how*
- Build intuition through experimentation
- Connect concepts across domains
- Ready for cutting-edge research

## 🆘 Getting Help & Contributing

### If You Get Stuck

1. **Check the docs**: Review `GETTING_STARTED_PLAN.md` and phase READMEs
2. **Read the code**: Utility modules have detailed comments
3. **Visualize**: Use plotting tools to understand behavior
4. **Experiment**: Try changing parameters to build intuition
5. **Document**: Write down your confusion - often solves itself!

### For Other Learners

- 💬 **Discussions**: Use GitHub Discussions for questions
- 🐛 **Issues**: Report bugs or unclear instructions
- ⭐ **Star**: If this helps you, star the repo!
- 🍴 **Fork**: Adapt for your learning style

### Contributing

Found a bug? Have an improvement? See `CONTRIBUTING.md` for the conventions.

```bash
# Fork the repo
# Create a branch
git checkout -b feature/your-improvement

# Make changes and test
pytest tests/

# Commit and push
git commit -m "Add: your improvement"
git push origin feature/your-improvement

# Open a Pull Request
```

**Good contributions**:
- Fixing errors in notebooks
- Adding visualizations
- Improving documentation
- Reporting broken cross-references between docs

## 🙏 Acknowledgments

**Inspired by**:
- Andrej Karpathy's "Neural Networks: Zero to Hero"
- Fast.ai's practical deep learning approach
- Stanford CS231n and CS224n courses
- The MLX community

**Built with**:
- NumPy, scikit-learn, PyTorch for implementations
- MLX for Apple Silicon optimization
- Jupyter for interactive learning
- Lots of ☕ and determination

## 📊 Learning Statistics

**Curriculum Stats**:
- 📚 21 hands-on projects (17 core + 4 bridges: 11.5, 11.75, 12.1, 12.25)
- 💻 21 Jupyter notebooks
- 🛠️ 5 utility modules (`utils/`)
- 📖 4 detailed learning-path markdown docs + glossary + getting-started
- ⏱️ ~180-220 hours of hands-on coding
- 🎓 ~4-5 months at recommended cadence

**Difficulty Progression**:
```
Difficulty
│
│                                           ╱─────── Phase 3
│                               ╱──────────╱        (Advanced)
│                   ╱──────────╱              + Professional Topics
│       ╱──────────╱          Phase 2                (Optional)
│  ────╱                      (Intermediate)
│  Phase 1
│  (Beginner)
└────────────────────────────────────────────────────────> Time (weeks)
   0        5        10       15       20       25       30
```

## 🗺️ Roadmap

**Current Status**: ✅ Complete curriculum (v1.0)

**Future Additions** (Community-driven):
- [ ] Video walkthroughs for each project
- [ ] Additional datasets and experiments
- [ ] Reinforcement Learning from Human Feedback (RLHF) module
- [ ] Distributed training examples
- [ ] More AI safety case studies
- [ ] Translation to other languages

**Want to contribute?** See contributing section above!

## 📈 Your Next Steps

### Today (15 minutes)
- [ ] Clone this repository
- [ ] Run `./scripts/setup_environment.sh`
- [ ] Open Project 1 notebook in Jupyter
- [ ] Read the theoretical foundation section

### This Week (10-15 hours)
- [ ] Complete Project 1: Linear Regression
- [ ] Experiment with different learning rates
- [ ] Note your insights (in your own notes)
- [ ] Start Project 2: Logistic Regression

### This Month (40-60 hours)
- [ ] Complete Projects 1-4 (Fundamentals)
- [ ] Build strong optimization intuition
- [ ] Understand loss functions deeply

### In 3 Months (Phase 1 Complete)
- [ ] Finish all classical ML projects (1-11) plus bridges (11.5, 11.75)
- [ ] Ready for transformer architecture

### In 5 Months (All Phases)
- [ ] Built transformer from scratch
- [ ] Pretrained your own model
- [ ] Fine-tuned Qwen2.5-1.5B-Instruct
- [ ] **Ready for ML research or engineering roles!**

## ⚠️ Important Reminders

### Do's ✅
- ✅ **Implement from scratch first** - Understand before optimizing
- ✅ **Visualize everything** - Plots reveal understanding
- ✅ **Experiment freely** - Break things to learn
- ✅ **Document insights** - Your future self will thank you
- ✅ **Take your time** - Deep learning requires deep understanding

### Don'ts ❌
- ❌ **Don't skip projects** - Each builds on previous ones
- ❌ **Don't rush** - Speed ≠ understanding
- ❌ **Don't copy-paste** - Type code to internalize
- ❌ **Don't skip visualization** - You'll miss key insights
- ❌ **Don't work in isolation** - Share your progress!

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

**TL;DR**: Free to use, modify, and share. Perfect for personal learning, classroom use, or building upon.

## 💬 Final Words

> "The best way to understand deep learning is to build it from scratch."
> 
> — This Curriculum

This journey takes months, not days. But when you finish, you won't just know *how* to use LLMs—you'll understand *why* they work.

You'll be able to:
- Read any ML paper and implement it
- Debug models by understanding internals  
- Design new architectures with confidence
- Contribute to cutting-edge research

**The goal isn't to finish fast. The goal is to understand deeply.**

Take your time. Enjoy the process. Build something amazing.

---

<div align="center">

**Ready to start your ML journey?**

[📚 Read Getting Started Guide](GETTING_STARTED_PLAN.md) | [🚀 Start Project 1](projects/phase1_classical_ml/project01_linear_regression/)

**Happy Learning! 🎓✨**

*Built with ❤️ for deep understanding*

</div>
