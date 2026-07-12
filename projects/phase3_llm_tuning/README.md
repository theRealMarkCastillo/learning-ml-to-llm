# Phase 3: LLM Fine-tuning

## Overview
Fine-tune Qwen 1.5B and analyze how instruction tuning changes model behavior.

## Projects (Weeks 19-22)

### Project 16: Qwen Instruction Tuning (Week 19-21)
Fine-tune Qwen 2.5 1.5B Instruct using MLX:
- LoRA parameter-efficient fine-tuning
- Instruction dataset preparation
- Training loop for production model
- Before/after evaluation
- **Memory**: ~4-6GB peak
- **Time**: Smoke test ~1 min (DRY_RUN=True); real run varies by steps
- Covers LoRA mechanics, rank sensitivity, and domain specialization exercises

### Project 17: Comparative Analysis (Week 21-22)
Systematic comparison of base vs tuned:
- Instruction-following quality
- General capability preservation
- Attention pattern differences
- Failure mode analysis
- Research-grade documentation
- ✨ Enhanced: Production-focused evaluation framework, capability preservation checks, out-of-domain generalization analysis, curriculum summary

## Learning Outcomes
- Hands-on LLM fine-tuning experience
- Understanding of LoRA and efficient training
- Rigorous model evaluation skills
- Connection to AI safety research

## Hardware Requirements
- Apple Silicon (M1+) with MLX: ✓ ideal, ~4-6GB peak
- Linux + CUDA GPU (8GB+ VRAM): ✓ works, similar profile
- Memory usage: ~4-6GB peak
- Training time: smoke test <1 min; real run duration depends on step count
- Backend auto-selected via `utils.device`

## Connection to Research
This phase directly supports AI safety research:
- Understanding how fine-tuning changes behavior
- Analyzing model responses to edge cases
- Documenting epistemic properties
- Systematic evaluation methodology

## Time Estimate
2-4 weeks

## Getting Started

1. **Review Project 14 & 15** - Refresh understanding of pretraining and evaluation
2. **Start Project 16** - Read intro on LoRA and quantization
3. **Prepare instruction data** - Format as specified in notebook
4. **Run fine-tuning** - Duration depends on step count; start with DRY_RUN=True
5. **Complete Project 17** - Evaluate systematically
6. **Document findings** - Ready for research or deployment

**Key Principle**: Fine-tuning is practical LLM customization. Careful evaluation ensures improvements without regression.
