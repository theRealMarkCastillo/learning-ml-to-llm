# Contributing

This is a personal learning repository. If you are reading this, you are
probably a learner working through it on your own, or someone curious
enough to look at the source.

## Learning from this repo

The intended audience is the repo's owner working through the projects
sequentially. If you are a different learner and find this useful:

1. Clone the repo locally and work through the notebooks in order
   (`projects/phase1_classical_ml/project01_linear_regression` first).
2. Run `./scripts/setup_environment.sh` then `source venv/bin/activate`.
3. Open the notebooks with Jupyter from the repo root.
4. Do not just read the cells — type them, run them, break them, fix them.

## Reporting issues

Issues are welcome if you spot:

- A factual error in a markdown doc or notebook markdown cell.
- A code cell that fails to execute on a clean install.
- A broken cross-reference between docs.
- A typo that meaningfully obscures meaning.

When opening an issue, please include:

- Which file (path + line number or notebook cell index)
- What you observed vs what you expected
- Your environment (Python version, OS, relevant package versions)

## Contributing code

Code contributions are not actively solicited. The repo is structured
to match one learner's path through the material. Substantial structural
changes are unlikely to be merged unless they fix a real bug or remove
a real error.

Small fixes (typos, broken imports, factual corrections) are welcome
via pull request.

## Style conventions

- Type hints throughout new Python.
- Google-style docstrings for public functions.
- No `print()` in library code; use `logging` or return values.
- Tests live in `tests/` and use `pytest`.
- Notebooks are the source of truth for project code; extracted
  `.py` modules exist only where the project explicitly factors them
  out (currently `projects/phase1_classical_ml/project11_5_neural_networks/neural_network.py`
  and `projects/phase2_transformers/project12_transformer_architecture/transformer.py`).

## License

By contributing, you agree your contributions are licensed under the
project's MIT license (see `LICENSE`).