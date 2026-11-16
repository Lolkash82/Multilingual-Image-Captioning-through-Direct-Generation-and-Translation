#!/bin/bash
# Environment A (OpenFlamingo). Run once on the login node. Adjust the torch wheel to the cluster CUDA.
set -euo pipefail
python3.11 -m venv venv-flamingo
source venv-flamingo/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install --no-deps git+https://github.com/mlfoundations/open_flamingo.git
pip freeze > environment-flamingo.lock.txt
python -c "import torch, transformers, open_flamingo; print(torch.__version__, transformers.__version__)"
