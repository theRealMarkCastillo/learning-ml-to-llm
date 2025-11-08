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
