# Quick Reference Checklist

## Key terms to know (developers new to ML)

See the full glossary at [docs/GLOSSARY.md](docs/GLOSSARY.md). Essentials you'll encounter immediately:

- Loss / Cross-Entropy / Perplexity: what you optimize and how you read progress
- Gradient / Learning Rate / Epoch / Batch: how updates happen
- Token / Tokenizer / Vocabulary: how text becomes model input
- Softmax / Sigmoid: how probabilities are produced
- LoRA / Adapter / Rank: how you fine-tune large models efficiently
- Attention / Head / Causal Mask: core transformer mechanics

## Repository Layout

The repo ships with these top-level files and directories:

| Path | What it is |
|------|------------|
| `README.md` | Landing page |
| `GETTING_STARTED_PLAN.md` | Day-1 setup walkthrough |
| `QUICK_REFERENCE.md` | This file |
| `CONTRIBUTING.md` | Conventions and bug reporting |
| `classical_ml_learning_path.md` | Phase 1 deep-dive |
| `complete_ml_learning_path_with_pretraining.md` | Phases 1-2 deep-dive |
| `qwen_mlx_learning_project.md` | Phase 3 deep-dive |
| `requirements.txt` | Python dependencies |
| `.gitignore` | What git ignores |
| `docs/GLOSSARY.md` | Terminology reference |
| `docs/LEARNING_OVERVIEW.md` | Strategy and milestones |
| `utils/` | Shared modules used by all notebooks |
| `scripts/` | Setup, downloader, verification scripts |
| `tests/` | pytest suite |
| `projects/phase1_classical_ml/` | Projects 1-11 + bridges 11.5, 11.75 |
| `projects/phase2_transformers/` | Bridges 12.1, 12.25 + projects 12-15 |
| `projects/phase3_llm_tuning/` | Projects 16, 17 |

## Projects (21 total)

**Phase 1: Classical ML** — 13 entries (11 core + 2 bridges)
- Project 1: Linear Regression
- Project 2: Logistic Regression
- Project 3: Multi-class Classification
- Project 4: Regularization
- Project 5: Decision Trees
- Project 6: Random Forests
- Project 7: Classification Metrics
- Project 8: Cross-Validation
- Project 9: SVMs
- Project 10: Feature Engineering
- Project 11: End-to-End Pipeline
- **Project 11.5 (bridge)**: Neural Networks from Scratch
- **Project 11.75 (bridge)**: RNNs from Scratch

**Phase 2: Transformers** — 6 entries (4 core + 2 bridges)
- **Project 12.1 (bridge)**: Attention Mechanisms
- **Project 12.25 (bridge)**: Embeddings from Scratch
- Project 12: Transformer Architecture
- Project 13: Tokenization
- Project 14: Pretraining (core centerpiece)
- Project 15: Pretrained vs Random Analysis

**Phase 3: LLM Fine-tuning** — 2 entries
- Project 16: Qwen2.5-1.5B Instruction Tuning (LoRA + MLX)
- Project 17: Base vs Tuned Comparative Analysis

## How to Use

### Option 1: Automated Setup (recommended)

```bash
cd <your-clone-path>/learning-ml-to-llm
./scripts/setup_environment.sh
source venv/bin/activate
jupyter notebook
```

### Option 2: Manual Setup

```bash
cd <your-clone-path>/learning-ml-to-llm
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m ipykernel install --user --name=ml-learning --display-name="Python (ML Learning)"
jupyter notebook
```

### Option 3: Verify the repo

```bash
./scripts/verify_repo.sh        # structural check
pytest tests/                    # run the test suite
python scripts/verify_device.py # confirm backend (MLX / CUDA / MPS / CPU)
```

## What to Do First

1. **Setup** (~15 minutes): run the setup script or manual steps; launch Jupyter.
2. **Start Project 1** (Days 1-5): open
   `projects/phase1_classical_ml/project01_linear_regression/linear_regression_from_scratch.ipynb`,
   execute every cell, run the experiments, attempt the exercises.
3. **Keep notes**: jot down what you learn (your own file, a journal, git commits — your call).
4. **Continue**: Phase 1 takes ~10-14 weeks; Phase 2 ~5 weeks; Phase 3 ~4 weeks.

## Utilities Ready to Use

```python
from utils.path_helpers import add_repo_root_to_sys_path
add_repo_root_to_sys_path()  # ensures `from utils...` works from any project subfolder

from utils.visualization import plot_loss_curve, plot_decision_boundary
from utils.data_generators import generate_linear_data, generate_binary_classification_data
from utils.metrics import mean_squared_error, accuracy, f1_score
from utils.device import backend_info, get_device, tensor, ensure_seed
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'utils'`

You're running from a subfolder without repo root on `sys.path`. Use the helper:

```python
from utils.path_helpers import add_repo_root_to_sys_path
add_repo_root_to_sys_path()
```

Or run the helper once in your notebook:

```python
import sys
from pathlib import Path

for candidate in [Path.cwd().resolve()] + list(Path.cwd().resolve().parents):
    if (candidate / "requirements.txt").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break
```

### Jupyter kernel not showing?

```bash
python3 -m ipykernel install --user --name=ml-learning --display-name="Python (ML Learning)"
```

### Need to reinstall?

```bash
rm -rf venv
./scripts/setup_environment.sh
```

### Verify the device backend (GPU / MPS / MLX / CPU)

```bash
python scripts/verify_device.py
# Optional override
LEARNING_ML_BACKEND=cpu python scripts/verify_device.py
```

If you're on Apple Silicon with MLX installed, you'll see `Backend=MLX ...`. On Linux with NVIDIA, `Backend=torch_cuda ...`. On Intel macOS without GPU, it falls back to CPU.

## Success Criteria

You're on track if:

- You can explain concepts without notes
- Your code works from scratch (not copy-paste)
- You can predict parameter changes
- You can debug using understanding
- You're documenting insights regularly

## Need Help?

1. Check [docs/GLOSSARY.md](docs/GLOSSARY.md) for terminology
2. Review [GETTING_STARTED_PLAN.md](GETTING_STARTED_PLAN.md) for setup
3. Read code comments — utility modules have detailed docstrings
4. Experiment — try changing parameters to learn
5. Write down your confusion — the act of writing often solves it

---

**The goal is deep understanding, not racing to the end.**

Happy learning.