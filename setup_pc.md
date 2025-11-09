# PC Setup Guide — Running AdaptCommand with Local GPU

This guide will help you set up and run the AdaptCommand benchmark on your PC with a local GPU.

## Step 1: Check Your GPU

### For NVIDIA GPUs (CUDA)

```bash
# Check if NVIDIA GPU is detected
nvidia-smi

# Check CUDA version
nvcc --version

# Alternative: Check CUDA from nvidia-smi
nvidia-smi | grep "CUDA Version"
```

Expected output from `nvidia-smi`:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.86.10    Driver Version: 535.86.10    CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... Off  | 00000000:01:00.0  On |                  N/A |
```

### For AMD GPUs (ROCm)

```bash
# Check AMD GPU
rocm-smi

# Check ROCm version
rocminfo | grep "ROCm"

# List available GPUs
rocminfo | grep "Device Type"
```

### For Intel GPUs

```bash
# Check Intel GPU (Linux)
clinfo

# Check Intel GPU (Windows with Intel drivers)
# Use Device Manager or:
wmic path win32_VideoController get name
```

### For CPU-only (Fallback)

```bash
# Check CPU info (Linux/Mac)
lscpu

# Check CPU cores
nproc

# Mac specific
sysctl -n hw.ncpu
```

---

## Step 2: Set Up Python Environment

### Check Python Version

```bash
# Check Python version (need 3.8+)
python --version
python3 --version

# If Python is not installed, download from python.org
# Recommended: Python 3.10 or 3.11
```

### Create Virtual Environment

```bash
# Navigate to project directory
cd /path/to/adaptcommand

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Verify activation (should show venv path)
which python
```

---

## Step 3: Install PyTorch with GPU Support

### For NVIDIA GPUs (CUDA)

```bash
# Check your CUDA version first (from Step 1)
nvidia-smi

# Install PyTorch with CUDA support
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For latest CUDA:
pip install torch torchvision torchaudio
```

### For AMD GPUs (ROCm)

```bash
# Install PyTorch with ROCm support (Linux only)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

### For CPU-only (No GPU)

```bash
# Install CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Verify PyTorch Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output (with GPU):
```
PyTorch: 2.1.0+cu118
CUDA available: True
CUDA version: 11.8
GPU count: 1
GPU name: NVIDIA GeForce RTX 3060
```

---

## Step 4: Install Project Dependencies

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install all dependencies
pip install -r requirements.txt

# Verify key packages
pip list | grep transformers
pip list | grep sentence-transformers
pip list | grep rouge-score
```

### If You Get Installation Errors

```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Retry installation
pip install -r requirements.txt

# If specific package fails, install individually
pip install transformers
pip install sentence-transformers
pip install rouge-score
pip install numpy pandas tqdm
```

---

## Step 5: Test GPU with Python

Create a test script to verify everything works:

```bash
# Create a test file
cat > test_gpu.py << 'EOF'
import torch
import transformers
from sentence_transformers import SentenceTransformer

print("="*60)
print("GPU SETUP TEST")
print("="*60)

# PyTorch
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")

    # Test tensor on GPU
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print(f"\n✓ GPU tensor test passed")
else:
    print("\n⚠ No CUDA GPU detected, will use CPU")

# Test transformers
print(f"\nTransformers version: {transformers.__version__}")

# Test sentence-transformers (small model)
print("\nTesting sentence-transformers...")
try:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    test_sentence = ["This is a test"]
    embedding = model.encode(test_sentence)
    print(f"✓ Sentence-transformers test passed")
    print(f"  Embedding shape: {embedding.shape}")
except Exception as e:
    print(f"✗ Sentence-transformers test failed: {e}")

print("\n" + "="*60)
print("SETUP COMPLETE")
print("="*60)
EOF

# Run the test
python test_gpu.py
```

---

## Step 6: Monitor GPU Usage During Training

### Real-time GPU Monitoring

```bash
# NVIDIA GPUs: Monitor in real-time (updates every 1 second)
watch -n 1 nvidia-smi

# Or simpler version
nvidia-smi -l 1

# AMD GPUs
watch -n 1 rocm-smi

# Check GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Monitor specific GPU
nvidia-smi -i 0  # GPU 0
```

### During Benchmark Run

Open a second terminal and run:
```bash
# Terminal 1: Run benchmark
python scripts/run_baseline_benchmark.py --config configs/baseline_v1.0.json --max_examples 50

# Terminal 2: Monitor GPU
nvidia-smi -l 1
```

---

## Step 7: Run Your First Benchmark

### Quick Test (50 examples, ~2-5 minutes)

```bash
# Activate environment
source venv/bin/activate

# Run quick test
python scripts/run_baseline_benchmark.py \
  --config configs/baseline_v1.0.json \
  --max_examples 50 \
  --output_dir results/baseline_quick
```

### Check Output

```bash
# View results
cat results/baseline_quick/baseline_results.json | python -m json.tool

# View sample outputs
head -n 20 results/baseline_quick/sample_outputs.jsonl
```

### Full Baseline Run (2000 examples, ~30-60 minutes)

```bash
# Make sure GPU is available
nvidia-smi

# Run full benchmark
python scripts/run_baseline_benchmark.py \
  --config configs/baseline_v1.0.json \
  --output_dir results/baseline
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

```bash
# Symptoms:
# RuntimeError: CUDA out of memory

# Solutions:
# 1. Use smaller model in config
# Edit configs/baseline_v1.0.json:
# "name": "Qwen/Qwen2.5-0.5B-Instruct"  # Instead of larger models

# 2. Reduce batch size (future enhancement)

# 3. Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# 4. Check what's using GPU
nvidia-smi
# Kill other processes using GPU if needed
```

### Issue: CUDA Not Available

```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False:
# 1. Check driver
nvidia-smi

# 2. Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Check CUDA installation
nvcc --version
```

### Issue: Slow Performance on GPU

```bash
# Check GPU utilization
nvidia-smi

# If GPU usage is low (< 50%):
# 1. Model might be too small for GPU
# 2. Data loading might be bottleneck
# 3. Try larger batch size (future feature)

# Profile GPU usage
python -c "import torch; print(torch.cuda.memory_summary())"
```

### Issue: Model Download Fails

```bash
# Symptoms:
# OSError: Can't load tokenizer/model

# Solutions:
# 1. Check internet connection

# 2. Clear Hugging Face cache
rm -rf ~/.cache/huggingface/

# 3. Set proxy if needed
export HF_ENDPOINT=https://hf-mirror.com  # Mirror

# 4. Pre-download model
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')"
```

---

## Performance Optimization Tips

### 1. Use Mixed Precision (Automatic)

The script automatically uses `torch.float16` for CUDA GPUs, which:
- Reduces memory usage by ~50%
- Speeds up computation by ~2x
- Maintains accuracy for inference

### 2. Choose Right Model Size

Based on your GPU memory:

| GPU Memory | Recommended Model | Parameter Count |
|------------|-------------------|-----------------|
| 4-6 GB     | Qwen2.5-0.5B     | 0.5B            |
| 6-8 GB     | Qwen2.5-1.5B     | 1.5B            |
| 8-12 GB    | Phi-2            | 2.7B            |
| 12-16 GB   | Qwen2.5-7B       | 7B              |
| 24+ GB     | Llama-3.2-7B     | 7B              |

### 3. Monitor Temperature

```bash
# Check GPU temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv

# If temperature > 80°C:
# - Improve case ventilation
# - Clean dust from GPU fans
# - Consider reducing clock speeds
```

---

## Next Steps

Once setup is complete:

1. ✓ Run quick test (50 examples)
2. Run full baseline evaluation (2000 examples)
3. Review results in `results/baseline/`
4. Move to Phase C: Train global adapters
5. Move to Phase D: Train personalized adapters

See `QUICKSTART.md` for detailed workflow.

---

## Quick Reference Commands

```bash
# Check GPU
nvidia-smi

# Activate environment
source venv/bin/activate

# Test GPU in Python
python -c "import torch; print(torch.cuda.is_available())"

# Monitor GPU while running
watch -n 1 nvidia-smi

# Run quick benchmark
python scripts/run_baseline_benchmark.py --config configs/baseline_v1.0.json --max_examples 50

# View results
cat results/baseline_quick/baseline_results.json | python -m json.tool
```
