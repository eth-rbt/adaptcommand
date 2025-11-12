"""
Test script to verify hybrid training setup before running full training.

This checks:
1. Unified model exists
2. Data files exist
3. PEFT library supports adapter stacking
4. Can load and stack adapters successfully
"""

import sys
from pathlib import Path
import torch

def check_unified_model(unified_path: Path):
    """Check if unified model exists and is valid."""
    print(f"\n1. Checking unified model at {unified_path}...")

    if not unified_path.exists():
        print(f"   [ERROR] Unified model not found at {unified_path}")
        print(f"   Run: python scripts/train_unified_lora.py")
        return False

    # Check for either .bin or .safetensors format
    config_file = unified_path / "adapter_config.json"
    bin_file = unified_path / "adapter_model.bin"
    safetensors_file = unified_path / "adapter_model.safetensors"

    if not config_file.exists():
        print(f"   [ERROR] Missing adapter_config.json in unified model")
        return False

    if not (bin_file.exists() or safetensors_file.exists()):
        print(f"   [ERROR] Missing adapter_model.bin or adapter_model.safetensors in unified model")
        return False

    print(f"   [OK] Unified model found with all required files")
    return True


def check_data_files():
    """Check if data files exist."""
    print(f"\n2. Checking data files...")

    dialogues = Path("data/cleaned/dialogs_clean.jsonl")
    splits = Path("data/splits/edgesplits.json")

    if not dialogues.exists():
        print(f"   [ERROR] Dialogues file not found at {dialogues}")
        return False

    if not splits.exists():
        print(f"   [ERROR] Splits file not found at {splits}")
        return False

    print(f"   [OK] Data files found")
    return True


def check_peft_version():
    """Check PEFT library version supports adapter stacking."""
    print(f"\n3. Checking PEFT library...")

    try:
        import peft
        from peft import PeftModel, LoraConfig

        version = peft.__version__
        print(f"   [OK] PEFT version: {version}")

        # Check if key methods exist
        if not hasattr(PeftModel, 'add_adapter'):
            print(f"   [WARNING] PEFT version may not support adapter stacking")
            print(f"   Consider upgrading: pip install -U peft")
            return False

        print(f"   [OK] PEFT supports adapter stacking")
        return True

    except ImportError:
        print(f"   [ERROR] PEFT not installed")
        print(f"   Install: pip install peft")
        return False


def test_adapter_stacking(unified_path: Path):
    """Test if we can stack adapters."""
    print(f"\n4. Testing adapter stacking...")

    try:
        from transformers import AutoModelForCausalLM
        from peft import PeftModel, LoraConfig, get_peft_model

        # Load tiny model for testing
        print(f"   Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct",
            dtype=torch.float32,
            device_map="cpu"
        )

        # Load unified adapter
        print(f"   Loading unified adapter...")
        model_with_unified = PeftModel.from_pretrained(
            base_model,
            str(unified_path),
            adapter_name="unified"
        )

        # Test adding a second adapter
        print(f"   Adding test persona adapter...")
        test_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model_with_unified.add_adapter("test_persona", test_config)

        # Verify both adapters exist
        print(f"   Verifying adapters...")
        if hasattr(model_with_unified, 'peft_config'):
            adapters = list(model_with_unified.peft_config.keys())
            print(f"   Found adapters: {adapters}")
            if 'unified' in adapters and 'test_persona' in adapters:
                print(f"   [OK] Adapter stacking works! Both adapters loaded successfully")
            else:
                print(f"   [WARNING] Not all adapters found")
                return False
        else:
            # Older API - just check it didn't error
            print(f"   [OK] Adapter stacking works!")

        del model_with_unified
        del base_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return True

    except Exception as e:
        print(f"   [ERROR] Adapter stacking failed: {e}")
        return False


def check_gpu():
    """Check GPU availability."""
    print(f"\n5. Checking GPU...")

    if torch.cuda.is_available():
        print(f"   [OK] CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        print(f"   [OK] MPS (Apple Silicon) available")
    else:
        print(f"   [WARNING] No GPU available, training will be slow")
        print(f"   Consider using CUDA GPU for faster training")

    return True


def main():
    print("="*60)
    print("HYBRID TRAINING SETUP TEST")
    print("="*60)

    unified_path = Path("models/lora_unified")

    checks = [
        check_unified_model(unified_path),
        check_data_files(),
        check_peft_version(),
        check_gpu()
    ]

    # Only test adapter stacking if other checks pass
    if all(checks[:3]):
        checks.append(test_adapter_stacking(unified_path))

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if all(checks):
        print("\n[OK] All checks passed!")
        print("\nYou can now run hybrid training:")
        print("  python scripts/train_persona_on_unified.py --persona_id persona_000")
        print("  python scripts/train_all_hybrid.py")
        return 0
    else:
        print("\n[ERROR] Some checks failed")
        print("\nPlease fix the issues above before running hybrid training")
        return 1


if __name__ == "__main__":
    sys.exit(main())
