#!/bin/bash
# Quick start script to train unified LoRA model on all training data

# Default: Train with persona descriptions included
echo "Starting unified LoRA training on all training data (6,000 dialogues)..."
echo "This will train ONE model on all 200 personas x 30 training dialogues each"
echo ""

python scripts/train_unified_lora.py \
    --config configs/lora_training.json \
    --output_dir models/lora_unified \
    "$@"

echo ""
echo "Training complete! Model saved to models/lora_unified"
echo ""
echo "To train WITHOUT persona descriptions, run:"
echo "  python scripts/train_unified_lora.py --no_persona"
echo ""
echo "To train WITHOUT validation evaluation, run:"
echo "  python scripts/train_unified_lora.py --no_val"
