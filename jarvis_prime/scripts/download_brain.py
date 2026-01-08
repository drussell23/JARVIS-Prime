#!/usr/bin/env python3
"""
JARVIS-Prime Brain Downloader
==============================

Downloads and sets up GGUF models for M1/Apple Silicon inference.

Usage:
    # Download recommended model (Llama 3 8B for M1)
    python -m jarvis_prime.scripts.download_brain

    # Download specific model
    python -m jarvis_prime.scripts.download_brain --model llama3-8b

    # List available models
    python -m jarvis_prime.scripts.download_brain --list

    # Download and verify
    python -m jarvis_prime.scripts.download_brain --model llama3-8b --verify
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print the JARVIS-Prime banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║     ╦╔═╗╦═╗╦  ╦╦╔═╗   ╔═╗╦═╗╦╔╦╗╔═╗                               ║
║     ║╠═╣╠╦╝╚╗╔╝║╚═╗───╠═╝╠╦╝║║║║║╣                                ║
║    ╚╝╩ ╩╩╚═ ╚╝ ╩╚═╝   ╩  ╩╚═╩╩ ╩╚═╝                               ║
║                                                                   ║
║                    BRAIN DOWNLOADER v80.0                         ║
║                                                                   ║
║         M1/M2/M3/M4 Optimized GGUF Model Installer                ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def list_available_models():
    """List all available preset models."""
    from jarvis_prime.core.llama_cpp_executor import RECOMMENDED_MODELS, HardwareDetector

    hw = HardwareDetector.detect()

    print("\n" + "=" * 70)
    print("AVAILABLE MODELS")
    print("=" * 70)
    print()

    print(f"Your Hardware: {hw.gpu_name or 'CPU'}")
    print(f"Total Memory: {hw.total_memory_gb:.1f} GB")
    print(f"Metal GPU: {'Enabled' if hw.metal_supported else 'Not Available'}")
    print()

    print("-" * 70)
    print(f"{'ID':<15} {'Size':<8} {'Template':<10} Description")
    print("-" * 70)

    for model_id, info in RECOMMENDED_MODELS.items():
        # Recommend based on memory
        recommend = ""
        if hw.metal_supported:
            if hw.total_memory_gb >= 16 and "8b" in model_id:
                recommend = " [RECOMMENDED]"
            elif 8 <= hw.total_memory_gb < 16 and info.size_gb <= 3:
                recommend = " [RECOMMENDED]"
            elif hw.total_memory_gb < 8 and "tiny" in model_id:
                recommend = " [RECOMMENDED]"

        print(f"{model_id:<15} {info.size_gb:<8.1f} {info.chat_template:<10} {info.description}{recommend}")

    print("-" * 70)
    print()


def list_local_models():
    """List locally downloaded models."""
    from jarvis_prime.core.llama_cpp_executor import GGUFModelDownloader

    downloader = GGUFModelDownloader()
    models = downloader.list_local_models()

    print("\n" + "=" * 70)
    print("LOCAL MODELS")
    print("=" * 70)

    if not models:
        print("\nNo local models found.")
        print(f"Models directory: {downloader.models_dir}")
        print("\nRun: python -m jarvis_prime.scripts.download_brain --model llama3-8b")
        return

    print()
    print("-" * 70)
    print(f"{'Name':<40} {'Size (GB)':<12} Path")
    print("-" * 70)

    for model in models:
        name = Path(model["path"]).name
        if len(name) > 38:
            name = name[:35] + "..."
        print(f"{name:<40} {model['size_gb']:<12.2f} {model['path']}")

    print("-" * 70)
    print(f"\nTotal: {len(models)} model(s)")
    print()


async def download_model(
    model_id: str,
    force: bool = False,
    verify: bool = False,
) -> Optional[Path]:
    """
    Download a model.

    Args:
        model_id: Model ID (preset or repo/filename)
        force: Force re-download
        verify: Verify model after download

    Returns:
        Path to downloaded model or None on failure
    """
    from jarvis_prime.core.llama_cpp_executor import (
        GGUFModelDownloader,
        RECOMMENDED_MODELS,
        LlamaCppExecutor,
        LlamaCppConfig,
    )

    downloader = GGUFModelDownloader()

    # Show download info
    if model_id in RECOMMENDED_MODELS:
        info = RECOMMENDED_MODELS[model_id]
        print(f"\nDownloading: {model_id}")
        print(f"  Repository: {info.repo_id}")
        print(f"  Filename: {info.filename}")
        print(f"  Size: ~{info.size_gb:.1f} GB")
        print(f"  Quantization: {info.quantization.value}")
        print(f"  Chat Template: {info.chat_template}")
        print()
    else:
        print(f"\nDownloading: {model_id}")
        print()

    # Progress callback
    def progress(filename: str, progress: float):
        bar_width = 40
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"\r  [{bar}] {progress*100:.1f}%", end="", flush=True)

    try:
        print("Starting download...")
        model_path = await downloader.download(
            model_id=model_id,
            progress_callback=progress,
            force=force,
        )
        print()  # New line after progress bar
        print(f"\n✓ Download complete: {model_path}")
        print(f"  Size: {model_path.stat().st_size / (1024**3):.2f} GB")

        # Verify if requested
        if verify:
            print("\nVerifying model...")
            config = LlamaCppConfig.auto_detect()
            executor = LlamaCppExecutor(config)

            try:
                await executor.load(model_path)
                is_valid = await executor.validate()
                await executor.unload()

                if is_valid:
                    print("✓ Model verified successfully!")
                else:
                    print("✗ Model validation failed!")
                    return None
            except Exception as e:
                print(f"✗ Model verification error: {e}")
                return None

        return model_path

    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return None


async def auto_download() -> Optional[Path]:
    """Auto-detect hardware and download recommended model."""
    from jarvis_prime.core.llama_cpp_executor import (
        GGUFModelDownloader,
        HardwareDetector,
    )

    hw = HardwareDetector.detect()
    downloader = GGUFModelDownloader()

    print("\nAuto-detecting recommended model for your hardware...")
    print(f"  Hardware: {hw.gpu_name or 'CPU'}")
    print(f"  Memory: {hw.total_memory_gb:.1f} GB")
    print(f"  Metal: {'Enabled' if hw.metal_supported else 'Disabled'}")

    recommended = downloader.get_recommended_model()

    if recommended is None:
        print("\n✗ No suitable model found for your hardware.")
        return None

    print(f"\nRecommended model: {recommended.description}")
    print(f"  Size: ~{recommended.size_gb:.1f} GB")

    # Confirm
    response = input("\nDownload this model? [Y/n]: ").strip().lower()
    if response in ("n", "no"):
        print("Download cancelled.")
        return None

    return await download_model(
        f"{recommended.repo_id}/{recommended.filename}",
        verify=True,
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="JARVIS-Prime Brain Downloader - Download GGUF models for M1 inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-download recommended model
    python -m jarvis_prime.scripts.download_brain

    # Download specific preset
    python -m jarvis_prime.scripts.download_brain --model llama3-8b

    # Download from HuggingFace
    python -m jarvis_prime.scripts.download_brain --model "TheBloke/Llama-2-7B-GGUF/llama-2-7b.Q4_K_M.gguf"

    # List available presets
    python -m jarvis_prime.scripts.download_brain --list

    # Show local models
    python -m jarvis_prime.scripts.download_brain --local
        """,
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Model ID to download (preset or repo/filename)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available preset models",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="List locally downloaded models",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if exists",
    )
    parser.add_argument(
        "--verify", "-v",
        action="store_true",
        help="Verify model after download",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode (minimal output)",
    )

    args = parser.parse_args()

    if not args.quiet:
        print_banner()

    # List available models
    if args.list:
        list_available_models()
        return

    # List local models
    if args.local:
        list_local_models()
        return

    # Download
    if args.model:
        result = asyncio.run(download_model(
            model_id=args.model,
            force=args.force,
            verify=args.verify,
        ))
    else:
        # Auto-download
        result = asyncio.run(auto_download())

    if result:
        print("\n" + "=" * 70)
        print("SUCCESS!")
        print("=" * 70)
        print(f"\nModel ready at: {result}")
        print("\nStart JARVIS-Prime with:")
        print("  python3 run_supervisor.py")
        print("\nOr test directly:")
        print("  python -c \"")
        print("    import asyncio")
        print("    from jarvis_prime.core.llama_cpp_executor import create_executor_with_model")
        print("    async def test():")
        print(f"        executor = await create_executor_with_model()")
        print("        print(await executor.generate('Hello!'))")
        print("    asyncio.run(test())")
        print("  \"")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
