"""
GPU Setup Test Script

Run this to verify your GPU setup is working correctly.
"""

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
