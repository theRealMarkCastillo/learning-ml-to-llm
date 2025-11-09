# Phase 2: Transformers & Pretraining

## Overview
Build and pretrain a tiny transformer to understand what happens inside base models.

## Bridge Projects (Prerequisites)

### Project 12.1: Attention Mechanisms ⭐
Isolate attention mechanism before full transformer:
- Scaled dot-product attention (Attention(Q,K,V) = softmax(QK^T/√d_k)V)
- Multi-head attention with parallel heads
- Causal masking for autoregressive generation
- Visualize attention weights to understand learned patterns
- Why attention has better gradient flow than RNNs
- ✨ Enhanced: Exercises on attention weight analysis, head specialization, gradient flow comparison

### Project 12.25: Embeddings & Representation Learning ⭐
Learn word/token embeddings via Word2Vec skip-gram:
- Skip-gram loss: predict context from target word
- Character-level embeddings on Shakespeare
- Visualization of learned semantic space (PCA 2D)
- Connection to transformer token embeddings
- Why embeddings are learned, not fixed one-hot
- ✨ Enhanced: Exercises on skip-gram variants, subword semantics, embedding arithmetic

## Main Projects (Weeks 13-17)

### Project 12: Transformer Architecture (Week 13-14)
Build decoder-only transformer from scratch:
- Multi-head self-attention
- Feed-forward networks
- Positional embeddings
- Stacking transformer blocks
- ✨ Enhanced: Full project intro covering architecture assembly, exercises on scaling and ablation studies

### Project 13: Tokenization (Week 14)
Understand text preprocessing:
- Character-level vs subword tokenization
- Byte-pair encoding (BPE)
- Special tokens and vocabulary
- Data loader creation
- ✨ Enhanced: Detailed intro, exercises comparing tokenization strategies and vocab trade-offs

### Project 14: Pretraining (Week 15-16) ⭐ CORE PROJECT
Pretrain tiny transformer on Shakespeare:
- Next-token prediction loss
- Training loop and optimization
- Loss dynamics observation
- Text generation from trained model
- **Runtime**: 4-12 hours on M4
- ✨ Enhanced: Production-grade intro covering training loops, checkpointing, scaling laws; exercises on convergence and data efficiency

### Project 15: Analysis (Week 17)
Compare pretrained vs random models:
- Transfer learning effectiveness
- Why pretraining matters
- Fine-tuning vs training from scratch
- ✨ Enhanced: Systematic evaluation framework with capability preservation checks and failure mode analysis

## Learning Outcomes
- Deep understanding of transformer architecture
- Experience with pretraining process
- Insight into what base models learn
- Foundation for understanding Mistral fine-tuning
- ✨ Hands-on evaluation and analysis skills

## Hardware Requirements
- M4 Mac with 64GB RAM: ✓ Perfect
- Training time: 4-12 hours for tiny model
- Memory usage: ~3GB during training

## Time Estimate
4-6 weeks of focused learning

## Getting Started

1. **Start with Project 12.1** - Understand attention in isolation
2. **Move to Project 12.25** - Learn embeddings via skip-gram
3. **Build Project 12** - Assemble full transformer
4. **Tokenize with Project 13** - Prepare text for training
5. **Pretrain with Project 14** - See language learning happen (the centerpiece!)
6. **Analyze with Project 15** - Understand what was learned

**Key Principle**: Each project builds on the previous. Don't skip the bridge projects—they're critical for intuition.
