#!/usr/bin/env python3
"""
JARVIS-Prime Model Downloader - Get Claude-3.5-Sonnet Equivalent Local Models
==============================================================================

Downloads recommended GGUF models from Hugging Face for local inference.

RECOMMENDED MODELS (Claude-3.5-Sonnet alternatives):

┌─────────────────────────────────────────────────────────────────────────────┐
│  Model              │ Size   │ Quality │ Speed  │ Best For                 │
├─────────────────────┼────────┼─────────┼────────┼──────────────────────────┤
│  Qwen 2.5 3B        │ 1.8GB  │ ⭐⭐⭐    │ 50t/s  │ Fast testing             │
│  Llama 3.2 3B       │ 2.0GB  │ ⭐⭐⭐    │ 45t/s  │ Fast, good quality       │
│  Phi-3 Mini 3.8B    │ 2.3GB  │ ⭐⭐⭐⭐   │ 40t/s  │ Efficient reasoning      │
│  Qwen 2.5 7B        │ 4.4GB  │ ⭐⭐⭐⭐   │ 25t/s  │ Best balance             │
│  Llama 3.1 8B       │ 4.7GB  │ ⭐⭐⭐⭐   │ 22t/s  │ General purpose          │
│  Qwen 2.5 14B       │ 8.5GB  │ ⭐⭐⭐⭐⭐  │ 12t/s  │ High quality             │
│  Qwen 2.5 32B       │ 19GB   │ ⭐⭐⭐⭐⭐  │ 5t/s   │ Closest to Sonnet        │
└─────────────────────────────────────────────────────────────────────────────┘

Usage:
    # List available models
    python3 scripts/download_recommended_models.py --list

    # Download recommended model (Qwen 2.5 7B)
    python3 scripts/download_recommended_models.py --download qwen-7b

    # Download specific model
    python3 scripts/download_recommended_models.py --download llama-8b

    # Set as current model
    python3 scripts/download_recommended_models.py --set-current qwen-7b
"""

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# MODEL CATALOG
# =============================================================================

MODELS_DIR = Path.home() / "Documents" / "repos" / "jarvis-prime" / "models"

# Hugging Face model URLs (GGUF quantized versions)
MODEL_CATALOG: Dict[str, Dict] = {
    "tinyllama-1b": {
        "name": "TinyLlama 1.1B Chat",
        "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "size_mb": 638,
        "quality": 2,
        "speed": 60,
        "description": "Ultra-fast, minimal quality - for testing only",
    },
    "qwen-3b": {
        "name": "Qwen 2.5 3B Instruct",
        "url": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "size_mb": 1800,
        "quality": 3,
        "speed": 50,
        "description": "Fast, good quality - great for quick responses",
    },
    "llama-3b": {
        "name": "Llama 3.2 3B Instruct",
        "url": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "filename": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "size_mb": 2000,
        "quality": 3,
        "speed": 45,
        "description": "Latest Llama, compact and capable",
    },
    "phi3-mini": {
        "name": "Phi-3 Mini 3.8B",
        "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
        "filename": "Phi-3-mini-4k-instruct-q4.gguf",
        "size_mb": 2300,
        "quality": 4,
        "speed": 40,
        "description": "Microsoft's efficient model, great reasoning",
    },
    "qwen-7b": {
        "name": "Qwen 2.5 7B Instruct",
        "url": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf",
        "filename": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "size_mb": 4400,
        "quality": 4,
        "speed": 25,
        "description": "★ RECOMMENDED - Best quality/speed balance",
    },
    "llama-8b": {
        "name": "Llama 3.1 8B Instruct",
        "url": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "filename": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "size_mb": 4700,
        "quality": 4,
        "speed": 22,
        "description": "Meta's workhorse, excellent general purpose",
    },
    "qwen-14b": {
        "name": "Qwen 2.5 14B Instruct",
        "url": "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m.gguf",
        "filename": "qwen2.5-14b-instruct-q4_k_m.gguf",
        "size_mb": 8500,
        "quality": 5,
        "speed": 12,
        "description": "High quality, needs 8GB+ RAM",
    },
    "qwen-32b": {
        "name": "Qwen 2.5 32B Instruct",
        "url": "https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-GGUF/resolve/main/qwen2.5-32b-instruct-q4_k_m.gguf",
        "filename": "qwen2.5-32b-instruct-q4_k_m.gguf",
        "size_mb": 19000,
        "quality": 5,
        "speed": 5,
        "description": "★ CLOSEST TO SONNET - needs 16GB+ RAM",
    },
}


def print_catalog():
    """Print available models."""
    print("\n" + "="*80)
    print("  JARVIS-Prime Recommended Local LLMs")
    print("  (Claude-3.5-Sonnet Equivalents)")
    print("="*80)

    print(f"\n{'Model ID':<15} {'Name':<25} {'Size':<8} {'Quality':<10} {'Speed':<10}")
    print("-"*80)

    for model_id, info in MODEL_CATALOG.items():
        quality_str = "⭐" * info["quality"]
        speed_str = f"~{info['speed']}t/s"
        size_str = f"{info['size_mb']}MB"

        # Check if downloaded
        model_path = MODELS_DIR / info["filename"]
        status = "✓" if model_path.exists() else " "

        print(f"{status} {model_id:<13} {info['name']:<25} {size_str:<8} {quality_str:<10} {speed_str:<10}")
        print(f"  └─ {info['description']}")
        print()

    print("\nRecommendations:")
    print("  • For testing/development: qwen-3b or phi3-mini")
    print("  • For best balance: qwen-7b (RECOMMENDED)")
    print("  • For highest quality: qwen-32b (closest to Claude-3.5-Sonnet)")
    print()


def download_model(model_id: str) -> bool:
    """Download a model from Hugging Face."""
    if model_id not in MODEL_CATALOG:
        print(f"❌ Unknown model: {model_id}")
        print(f"   Available: {', '.join(MODEL_CATALOG.keys())}")
        return False

    info = MODEL_CATALOG[model_id]
    model_path = MODELS_DIR / info["filename"]

    # Check if already downloaded
    if model_path.exists():
        print(f"✓ Model already downloaded: {model_path}")
        return True

    # Ensure directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n📥 Downloading {info['name']}...")
    print(f"   URL: {info['url']}")
    print(f"   Size: {info['size_mb']}MB")
    print(f"   Destination: {model_path}")
    print()

    # Download with progress
    try:
        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 // total_size)
            bar_len = 40
            filled = int(bar_len * percent // 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r   [{bar}] {percent}% ({mb_downloaded:.0f}/{mb_total:.0f}MB)")
            sys.stdout.flush()

        urllib.request.urlretrieve(info["url"], model_path, progress_hook)
        print("\n\n✓ Download complete!")
        return True

    except Exception as e:
        print(f"\n\n❌ Download failed: {e}")
        if model_path.exists():
            model_path.unlink()
        return False


def set_current_model(model_id: str) -> bool:
    """Set a model as the current default."""
    if model_id not in MODEL_CATALOG:
        print(f"❌ Unknown model: {model_id}")
        return False

    info = MODEL_CATALOG[model_id]
    model_path = MODELS_DIR / info["filename"]
    current_link = MODELS_DIR / "current.gguf"

    if not model_path.exists():
        print(f"❌ Model not downloaded: {model_path}")
        print(f"   Run: python3 {sys.argv[0]} --download {model_id}")
        return False

    # Remove existing symlink
    if current_link.exists() or current_link.is_symlink():
        current_link.unlink()

    # Create new symlink
    current_link.symlink_to(info["filename"])
    print(f"✓ Set {info['name']} as current model")
    print(f"   Symlink: {current_link} -> {info['filename']}")
    return True


def show_current_model():
    """Show the currently selected model."""
    current_link = MODELS_DIR / "current.gguf"

    if not current_link.exists():
        print("No current model set")
        return

    if current_link.is_symlink():
        target = current_link.resolve()
        print(f"Current model: {target.name}")
        print(f"   Path: {target}")
        if target.exists():
            size_mb = target.stat().st_size / (1024 * 1024)
            print(f"   Size: {size_mb:.0f}MB")
    else:
        print(f"Current model: {current_link}")


def main():
    parser = argparse.ArgumentParser(
        description="Download recommended local LLMs for JARVIS-Prime"
    )
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--download", type=str, help="Download a model (e.g., qwen-7b)")
    parser.add_argument("--set-current", type=str, help="Set model as current default")
    parser.add_argument("--current", action="store_true", help="Show current model")

    args = parser.parse_args()

    if args.list:
        print_catalog()
    elif args.download:
        success = download_model(args.download)
        if success and args.set_current is None:
            # Ask if they want to set as current
            response = input("\nSet as current model? [Y/n]: ").strip().lower()
            if response in ("", "y", "yes"):
                set_current_model(args.download)
    elif args.set_current:
        set_current_model(args.set_current)
    elif args.current:
        show_current_model()
    else:
        # Default: show catalog
        print_catalog()
        print("\nUsage:")
        print(f"  python3 {sys.argv[0]} --list              # List available models")
        print(f"  python3 {sys.argv[0]} --download qwen-7b  # Download recommended model")
        print(f"  python3 {sys.argv[0]} --set-current qwen-7b  # Set as default")
        print(f"  python3 {sys.argv[0]} --current           # Show current model")


if __name__ == "__main__":
    main()
