#!/bin/bash

# Find the transformers installation path
TRANSFORMERS_PATH=$(python -c "import transformers; print(transformers.__path__[0])")

# Copy custom_trainer.py to the transformers directory, renaming it to trainer.py
cp custom_trainer.py "$TRANSFORMERS_PATH/trainer.py"

echo "custom_trainer.py has been copied to $TRANSFORMERS_PATH/trainer.py"
