#!/bin/bash
# Verification Script - Run this to verify everything is working

echo "================================================"
echo "ML Learning Repository Verification"
echo "================================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
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
check_file "PROGRESS_LOG.md"
check_file "CODE_REVIEW.md"
check_file "CONTRIBUTING.md"
check_file "LICENSE"

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

echo ""
echo "Checking Scripts..."
check_file "scripts/setup_environment.sh"
check_file "scripts/download_shakespeare.py"

echo ""
echo "Checking Data Directories..."
check_dir "data/raw"
check_dir "data/processed"

echo ""
echo "Checking Project Structure..."
echo "Phase 1: Classical ML (11 projects)"
for i in {1..11}; do
    check_dir "projects/phase1_classical_ml/project$(printf "%02d" $i)*"
done

echo ""
echo "Phase 2: Transformers (4 projects)"
for i in {12..15}; do
    check_dir "projects/phase2_transformers/project$(printf "%02d" $i)*"
done

echo ""
echo "Phase 3: LLM Fine-tuning (2 projects)"
for i in {16..17}; do
    check_dir "projects/phase3_llm_tuning/project$(printf "%02d" $i)*"
done

echo ""
echo "Checking Notebooks..."
NOTEBOOK_COUNT=$(find . -name "*.ipynb" | wc -l)
echo -e "${GREEN}✓${NC} Found $NOTEBOOK_COUNT notebooks"

echo ""
echo "Checking Python Files..."
PY_COUNT=$(find . -name "*.py" -type f ! -path "./venv/*" | wc -l)
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
echo "Happy learning! 🚀"
