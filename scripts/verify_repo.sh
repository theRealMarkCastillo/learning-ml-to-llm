#!/bin/bash
# Verification Script - Run this to verify everything is working

echo "================================================"
echo "ML Learning Repository Verification"
echo "================================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if files exist
check_file() {
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "${RED}✗${NC} $1"
        return 1
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "${RED}✗${NC} $1"
        return 1
    fi
}

echo "Checking Documentation Files..."
check_file "README.md"
check_file "GETTING_STARTED_PLAN.md"
check_file "QUICK_REFERENCE.md"
check_file "CONTRIBUTING.md"
check_file "LICENSE"

echo ""
echo "Checking Learning Paths..."
check_file "classical_ml_learning_path.md"
check_file "complete_ml_learning_path_with_pretraining.md"
check_file "qwen_mlx_learning_project.md"

echo ""
echo "Checking Configuration Files..."
check_file "requirements.txt"
check_file ".gitignore"

echo ""
echo "Checking Utility Modules..."
check_file "utils/__init__.py"
check_file "utils/visualization.py"
check_file "utils/data_generators.py"
check_file "utils/metrics.py"
check_file "utils/device.py"
check_file "utils/path_helpers.py"

echo ""
echo "Checking Docs..."
check_file "docs/GLOSSARY.md"
check_file "docs/LEARNING_OVERVIEW.md"

echo ""
echo "Checking Scripts..."
check_file "scripts/setup_environment.sh"
check_file "scripts/download_shakespeare.py"
check_file "scripts/verify_device.py"
check_file "scripts/verify_repo.sh"

echo ""
echo "Checking Data Directories..."
check_dir "data/raw"
check_dir "data/processed"

echo ""
echo "Checking Project Structure..."
echo "Phase 1: Classical ML (13 projects)"
for dir in \
    projects/phase1_classical_ml/project01_linear_regression \
    projects/phase1_classical_ml/project02_logistic_regression \
    projects/phase1_classical_ml/project03_multiclass \
    projects/phase1_classical_ml/project04_regularization \
    projects/phase1_classical_ml/project05_decision_trees \
    projects/phase1_classical_ml/project06_random_forests \
    projects/phase1_classical_ml/project07_classification_metrics \
    projects/phase1_classical_ml/project08_cross_validation \
    projects/phase1_classical_ml/project09_svm \
    projects/phase1_classical_ml/project10_feature_engineering \
    projects/phase1_classical_ml/project11_ml_pipeline \
    projects/phase1_classical_ml/project11_5_neural_networks \
    projects/phase1_classical_ml/project11_75_rnns; do
    check_dir "$dir"
done

echo ""
echo "Phase 2: Transformers (6 projects)"
for dir in \
    projects/phase2_transformers/project12_1_attention_mechanisms \
    projects/phase2_transformers/project12_25_embeddings \
    projects/phase2_transformers/project12_transformer_architecture \
    projects/phase2_transformers/project13_tokenization \
    projects/phase2_transformers/project14_pretraining \
    projects/phase2_transformers/project15_analysis; do
    check_dir "$dir"
done

echo ""
echo "Phase 3: LLM Fine-tuning (2 projects)"
for dir in \
    projects/phase3_llm_tuning/project16_qwen_tuning \
    projects/phase3_llm_tuning/project17_comparative_analysis; do
    check_dir "$dir"
done

echo ""
echo "Checking Tests..."
for f in tests/test_data_generators.py tests/test_device.py tests/test_metrics.py \
         tests/test_neural_network.py tests/test_path_helpers.py \
         tests/test_transformer.py tests/test_visualization.py; do
    check_file "$f"
done

echo ""
echo "Checking Notebooks..."
NOTEBOOK_COUNT=$(find . -name "*.ipynb" | wc -l | tr -d ' ')
echo -e "${GREEN}✓${NC} Found $NOTEBOOK_COUNT notebooks"

echo ""
echo "Checking Python Files..."
PY_COUNT=$(find . -name "*.py" -type f ! -path "./venv/*" | wc -l | tr -d ' ')
echo -e "${GREEN}✓${NC} Found $PY_COUNT Python files"

echo ""
echo "================================================"
echo "Verification Complete!"
echo "================================================"
echo ""
echo "To get started:"
echo "  1. ./scripts/setup_environment.sh"
echo "  2. source venv/bin/activate"
echo "  3. jupyter notebook"
echo ""
echo "Happy learning!"