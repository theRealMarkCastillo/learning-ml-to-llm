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

### Phase 1: Classical ML Foundation (Weeks 1-14)
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

### Phase 2: Transformers & Pretraining (Weeks 13-20)
**Goal**: Build and pretrain a transformer to understand base models

**Bridge Projects** (build intuition before assembly):
- 12.1: Attention Mechanisms from scratch
- 12.25: Embeddings & representation learning via skip-gram

**Core Projects**:
- Build transformer architecture from scratch
- Tokenization and text preprocessing
- **Pretrain tiny transformer on Shakespeare** (4-12 hours on M4)
- Analyze pretrained vs random models

**Key Learning**: Self-attention, multi-head attention, embeddings, pretraining dynamics, why base models work

[→ Phase 2 Details](projects/phase2_transformers/README.md)

### Phase 3: LLM Fine-tuning (Weeks 18-23)
**Goal**: Fine-tune Mistral 7B and analyze behavior changes

**Projects**:
- Instruction tune Mistral 7B with LoRA (using MLX)
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
- **4-8GB RAM** minimum (64GB recommended for Phase 3)
- **Jupyter** for running notebooks
- **Time commitment**: 10-20 hours/week for 4-6 months
- **Math background**: Basic calculus and linear algebra helpful but not required

### 1. Clone and Setup

```bash
# Clone this repository
git clone https://github.com/yourusername/learning-ml-to-llm.git
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
3. Document learnings in `PROGRESS_LOG.md`
4. Move to next project

**🎉 You're ready to start learning!**

## Repository Structure

```
learning-ml-to-llm/
├── projects/
│   ├── phase1_classical_ml/       # Projects 1-11 (with 4.5, 6.5, 7.5, 7.8)
│   ├── phase2_transformers/       # Projects 12-15 (with 12.5, 13.5)
│   └── phase3_llm_tuning/        # Projects 16-17
├── docs/
│   ├── GLOSSARY.md               # ML terminology reference
│   ├── LEARNING_OVERVIEW.md      # Learning strategy guide
│   ├── TESTING_GUIDE.md          # ML testing patterns & practices
│   ├── MLOPS_PROFESSIONAL_GUIDE.md      # Experiment tracking, A/B testing, monitoring
│   ├── RESPONSIBLE_AI_GUIDE.md   # Bias, fairness, explainability, privacy
│   └── PROFESSIONAL_TOPICS_OVERVIEW.md  # Integration & timeline
├── utils/
│   ├── visualization.py          # Plotting utilities
│   ├── data_generators.py        # Data generation
│   └── metrics.py                # Evaluation metrics
├── data/
│   ├── raw/                      # Raw datasets
│   └── processed/                # Processed data
├── scripts/
│   ├── setup_environment.sh      # Setup script
│   └── download_shakespeare.py   # Download training data
├── requirements.txt              # Python dependencies
├── GETTING_STARTED_PLAN.md      # Detailed getting started guide
├── PROGRESS_LOG.md              # Learning progress tracking
└── README.md                    # This file
```

## 💻 Hardware Requirements

### Minimum Specs (Phases 1-2)
- **CPU**: Any modern processor
- **RAM**: 4-8GB
- **Storage**: 5GB
- **OS**: macOS, Linux, or Windows

### Recommended Specs (All Phases)
- **CPU**: Apple Silicon (M1/M2/M3/M4) or modern x86
- **RAM**: 32-64GB for Phase 3 (Mistral fine-tuning)
- **Storage**: 20GB
- **OS**: macOS (for MLX optimization) or Linux

### What Runs Where

| Phase | Project | Time | RAM Needed |
|-------|---------|------|------------|
| Phase 1 | Classical ML (1-11) | Seconds-Minutes | 2-4GB |
| Phase 2 | Build Transformer (12-13) | Minutes | 1-2GB |
| Phase 2 | Pretrain Tiny Model (14) | 4-12 hours | 3-8GB |
| Phase 3 | Fine-tune Mistral (16) | Hours | 20-30GB |

**Good news**: Phases 1-2 run on any laptop. Only Phase 3 needs serious hardware.

**Apple Silicon users**: MLX makes your Mac perfect for all phases!

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
- **Total timeline**: 19-23 weeks (~4-6 months)

## Progress Tracking

Track your progress in `PROGRESS_LOG.md`:
- Projects completed
- Key insights learned
- Challenges encountered
- Experimental results

## Resources

### Included Documents

**Learning Paths**:
- `classical_ml_learning_path.md` - Detailed Phase 1 guide (Projects 1-11)
- `complete_ml_learning_path_with_pretraining.md` - Full journey (Projects 12-17)
- `mistral_mlx_learning_project.md` - MLX fine-tuning guide
- `GETTING_STARTED_PLAN.md` - Step-by-step setup

**Professional Development Guides** (in `docs/`):
- `TESTING_GUIDE.md` - ML testing patterns, pytest, CI/CD
- `MLOPS_PROFESSIONAL_GUIDE.md` - Experiment tracking, A/B testing, monitoring
- `RESPONSIBLE_AI_GUIDE.md` - Bias detection, fairness, explainability, privacy
- `PROFESSIONAL_TOPICS_OVERVIEW.md` - Integration guide and timeline

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
model = AutoModel.from_pretrained("mistral-7b")
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

Found a bug? Have an improvement? Contributions welcome!

```bash
# Fork the repo
# Create a branch
git checkout -b feature/your-improvement

# Make changes and test
# Commit and push
git commit -m "Add: your improvement"
git push origin feature/your-improvement

# Open a Pull Request
```

**Good contributions**:
- Fixing errors in notebooks
- Adding visualizations
- Improving documentation
- Adding new experiments
- Sharing your learning insights

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
- 📚 20+ comprehensive projects (17 core + 4 professional extension projects)
- 💻 20+ Jupyter notebooks
- 🛠️ 3 utility modules with 30+ functions
- 📖 10 detailed documentation files (6 learning paths + 4 professional guides)
- ⏱️ ~200-250 hours of hands-on coding (150-200 core + 40-60 professional)
- 🎓 4-7 months total learning time (19-23 weeks core + 4-7 weeks professional topics)

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
- [ ] Document insights in `PROGRESS_LOG.md`
- [ ] Start Project 2: Logistic Regression

### This Month (40-60 hours)
- [ ] Complete Projects 1-4 (Fundamentals)
- [ ] Build strong optimization intuition
- [ ] Understand loss functions deeply

### In 3 Months (Phase 1 Complete)
- [ ] Finish all classical ML projects (1-11)
- [ ] Ready for transformer architecture

### In 6 Months (All Phases)
- [ ] Built transformer from scratch
- [ ] Pretrained your own model
- [ ] Fine-tuned Mistral 7B
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

[📚 Read Getting Started Guide](GETTING_STARTED_PLAN.md) | [🚀 Start Project 1](projects/phase1_classical_ml/project01_linear_regression/) | [⭐ Star This Repo](../../stargazers)

**Happy Learning! 🎓✨**

*Built with ❤️ for deep understanding*

</div>
