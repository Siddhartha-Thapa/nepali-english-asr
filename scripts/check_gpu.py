# Create: scripts/check_gpu.py
import torch
import sys

def check_gpu():
    print("=" * 60)
    print("GPU SETUP VERIFICATION")
    print("=" * 60)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\n✓ CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ GPU Device Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("\n⚠️  WARNING: No GPU detected. Training will be very slow!")
        print("   Consider using Google Colab or cloud GPU services.")
    
    # Check PyTorch version
    print(f"\n✓ PyTorch Version: {torch.__version__}")
    print("=" * 60)

if __name__ == "__main__":
    check_gpu()