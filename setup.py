#!/usr/bin/env python3
"""
Setup script for LLADA-8B-Instruct analysis pipeline.
Handles GPU checks, dependency installation, and HuggingFace authentication.
"""

import subprocess
import sys
import os
from pathlib import Path
from getpass import getpass

def print_header(text):
    print("\n" + "═" * 60)
    print(text)
    print("═" * 60)

def check_gpu():
    """Check GPU availability and display info."""
    print("\n▶ Checking GPU...")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print("  " + result.stdout.strip())
        else:
            print("  ⚠️  nvidia-smi not found")
    except Exception as e:
        print(f"  ⚠️  GPU check failed: {e}")

def check_pytorch():
    """Check PyTorch and CUDA availability."""
    print("\n▶ Checking PyTorch...")
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except ImportError:
        print("  ⚠️  PyTorch not installed — will be installed with dependencies")

def install_dependencies():
    """Install Python dependencies from requirements.txt."""
    print("\n▶ Installing Python dependencies...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        )
        print("  ✓ All dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to install dependencies: {e}")
        return False

def setup_hf_token():
    """Setup HuggingFace token authentication."""
    print("\n▶ Setting up HuggingFace authentication...")

    # Check if token already in environment
    token = os.environ.get("HF_TOKEN", "").strip()

    if token:
        print("  HF_TOKEN found in environment")
        return verify_hf_token(token)

    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        try:
            with open(env_file) as f:
                for line in f:
                    if line.startswith("HF_TOKEN="):
                        token = line.split("=", 1)[1].strip()
                        if token:
                            os.environ["HF_TOKEN"] = token
                            print("  HF_TOKEN loaded from .env")
                            return verify_hf_token(token)
        except Exception as e:
            print(f"  ⚠️  Failed to read .env: {e}")

    # Prompt user for token
    print("\n  No HF_TOKEN found. Get one from: https://huggingface.co/settings/tokens")
    print("  (Create a token with 'read' permission)\n")

    token = getpass("  Enter your HF_TOKEN: ").strip()
    if not token:
        print("  ✗ No token provided")
        return False

    os.environ["HF_TOKEN"] = token

    # Optionally save to .env
    if not env_file.exists():
        try:
            save = input("  Save token to .env? (y/n): ").strip().lower() == "y"
            if save:
                with open(env_file, "w") as f:
                    f.write(f"HF_TOKEN={token}\n")
                print("  ✓ Token saved to .env")
        except Exception as e:
            print(f"  ⚠️  Could not save to .env: {e}")

    return verify_hf_token(token)

def verify_hf_token(token):
    """Verify HuggingFace token is valid."""
    try:
        from huggingface_hub import login, whoami
        login(token=token, add_to_git_credential=False)
        user = whoami()
        print(f"  ✓ Authenticated as: {user['name']}")
        return True
    except Exception as e:
        print(f"  ✗ Authentication failed: {e}")
        return False

def setup_google_drive():
    """Setup Google Drive mounting (Colab only)."""
    print("\n▶ Checking for Google Drive (Colab)...")
    try:
        from google.colab import drive
        print("  Colab detected — mounting Google Drive for checkpoints...")

        # Check if already mounted
        gdrive_path = Path("/content/gdrive/My Drive")
        if gdrive_path.exists():
            print("  ✓ Google Drive already mounted")
            return True

        try:
            drive.mount("/content/gdrive", force_remount=True)
            # Wait a moment for mount to be ready
            import time
            time.sleep(2)

            if gdrive_path.exists():
                print("  ✓ Google Drive mounted at /content/gdrive")
                print("  ✓ Checkpoints will auto-save to: /My Drive/mdlm_checkpoint/")

                # Create checkpoint directory
                checkpoint_dir = gdrive_path / "mdlm_checkpoint"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                print(f"  ✓ Created checkpoint directory")
                return True
            else:
                print("  ⚠️  Mount point exists but /My Drive not accessible")
                return False

        except Exception as e:
            # Common error if kernel not fully initialized
            if "NoneType" in str(e) or "kernel" in str(e).lower():
                print("  ℹ️  Could not mount now (kernel initializing)")
                print("     Authorization dialog may appear — please authorize")
                return None
            else:
                print(f"  ⚠️  Could not mount Google Drive: {e}")
                print("     You can authorize manually with:")
                print("     from google.colab import drive; drive.mount('/content/gdrive', force_remount=True)")
                return False
    except ImportError:
        # Not in Colab
        return None

def main():
    """Run full setup."""
    print_header("LLADA-8B-Instruct Setup")

    # Check system
    check_gpu()
    check_pytorch()

    # Install dependencies
    if not install_dependencies():
        sys.exit(1)

    # Setup HuggingFace
    if not setup_hf_token():
        print("\n✗ Setup failed — could not authenticate with HuggingFace")
        sys.exit(1)

    # Setup Google Drive (Colab only)
    gdrive_result = setup_google_drive()
    if gdrive_result is None:
        # Either not in Colab, or kernel not ready yet
        pass
    elif not gdrive_result:
        print("\n⚠️  Google Drive mounting failed")
        print("   You can authorize it manually later:")

    print_header("✓ Setup Complete!")
    print("\nTo run the analysis:")
    print("  python full.py\n")

if __name__ == "__main__":
    main()
