#!/bin/bash
# Setup script for ML learning environment

set -e

echo "================================================"
echo "ML Learning Environment Setup"
echo "================================================"
echo ""

# Check Python version
echo "Checking Python installation..."
python3 --version

# Require Python 3.8+
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
MIN_VERSION="3.8"
if python3 -c "import sys; exit(0 if sys.version_info >= (3,8) else 1)" 2>/dev/null; then
    echo "  Python $PYTHON_VERSION (OK, >= $MIN_VERSION)"
else
    echo "  ERROR: Python $MIN_VERSION+ required (found $PYTHON_VERSION)"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install core packages
echo ""
echo "Installing core packages..."
pip install -r requirements.txt

# Optional: Install MLX on Apple Silicon (macOS arm64)
UNAME_S=$(uname -s || echo "")
UNAME_M=$(uname -m || echo "")
if [ "$UNAME_S" = "Darwin" ] && [ "$UNAME_M" = "arm64" ]; then
	echo "Detected macOS arm64 (Apple Silicon). Installing MLX packages..."
	# MLX projects are moving fast; allow failures without breaking setup
	pip install --upgrade mlx mlx-lm || echo "(warning) MLX install failed; continuing without MLX"
else
	echo "Non-Apple-Silicon platform detected. Skipping MLX packages."
fi

# Setup Jupyter kernel
echo ""
echo "Setting up Jupyter kernel..."
python3 -m ipykernel install --user --name=ml-learning --display-name="Python (ML Learning)"

# Create project directories
echo ""
echo "Creating project directories..."
mkdir -p data/raw data/processed
mkdir -p logs
mkdir -p checkpoints

echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start Jupyter:"
echo "  jupyter notebook"
echo ""
echo "Happy learning!"
