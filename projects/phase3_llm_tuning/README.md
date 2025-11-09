# Phase 3: LLM Fine-tuning

## Overview
Fine-tune Mistral 7B and analyze how instruction tuning changes model behavior.

## Projects (Weeks 18-23)

### Project 16: Mistral Instruction Tuning (Week 18-21)
Fine-tune real Mistral 7B using MLX:
- LoRA parameter-efficient fine-tuning
- Instruction dataset preparation
- Training loop for production model
- Before/after evaluation
- **Memory**: ~20-30GB on your M4
- **Time**: Hours per training run
- ✨ Enhanced: Comprehensive intro covering LoRA mechanics, quantization, and economics; exercises on rank sensitivity and domain specialization

### Project 17: Comparative Analysis (Week 22-23)
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
- ✨ Production-ready deployment understanding

## Hardware Requirements
- M4 Mac with 64GB RAM: ✓ Perfect
- Memory usage: ~20-30GB peak
- Training time: Hours per run
- MLX optimized for Apple silicon

## Connection to Research
This phase directly supports AI safety research:
- Understanding how fine-tuning changes behavior
- Analyzing model responses to edge cases
- Documenting epistemic properties
- Systematic evaluation methodology
- ✨ Rigorous capability preservation verification

## Time Estimate
4-6 weeks of intensive work

## Getting Started

1. **Review Project 14 & 15** - Refresh understanding of pretraining and evaluation
2. **Start Project 16** - Read intro on LoRA and quantization
3. **Prepare instruction data** - Format as specified in notebook
4. **Run fine-tuning** - Hours on M4, monitor for convergence
5. **Complete Project 17** - Evaluate systematically
6. **Document findings** - Ready for research or deployment

**Key Principle**: Fine-tuning is practical LLM customization. Careful evaluation ensures improvements without regression.
