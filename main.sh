#!/bin/bash

# in order to be able to run the script first run "chmod +x setup.sh"

source .venv/bin/activate

export TORCH_COMPILE_DEBUG=1

python train_llm_pytorch.py
