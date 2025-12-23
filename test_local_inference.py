#!/usr/bin/env python3
"""
Test local JARVIS-Prime inference with TinyLlama.
Run: python3 test_local_inference.py
"""
import sys
import time
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "current.gguf"

def test_basic_inference():
    """Test basic inference with TinyLlama."""
    print("=" * 60)
    print("JARVIS-Prime Local Inference Test")
    print("=" * 60)

    # Check model exists
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)

    model_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"Model: {MODEL_PATH.name}")
    print(f"Size: {model_size_mb:.1f} MB")
    print()

    # Import llama-cpp-python
    try:
        from llama_cpp import Llama
        print("llama-cpp-python: Installed")
    except ImportError:
        print("ERROR: llama-cpp-python not installed!")
        print("Run: pip3 install --user llama-cpp-python")
        sys.exit(1)

    # Load model
    print("\nLoading model...")
    start = time.time()

    llm = Llama(
        model_path=str(MODEL_PATH),
        n_ctx=2048,          # Context window
        n_threads=4,         # CPU threads
        n_gpu_layers=0,      # CPU only (set >0 for Metal GPU)
        verbose=False,
    )

    load_time = time.time() - start
    print(f"Model loaded in {load_time:.2f}s")

    # Test inference
    print("\n" + "-" * 60)
    print("Test 1: Simple completion")
    print("-" * 60)

    prompt = "Q: What is the capital of France?\nA:"
    print(f"Prompt: {prompt}")

    start = time.time()
    output = llm(
        prompt,
        max_tokens=32,
        temperature=0.1,
        stop=["\n", "Q:"],
        echo=False,
    )
    inference_time = time.time() - start

    response = output["choices"][0]["text"].strip()
    tokens_generated = output["usage"]["completion_tokens"]

    print(f"Response: {response}")
    print(f"Time: {inference_time:.2f}s ({tokens_generated / inference_time:.1f} tokens/sec)")

    # Test chat format
    print("\n" + "-" * 60)
    print("Test 2: Chat format (TinyLlama style)")
    print("-" * 60)

    chat_prompt = """<|system|>
You are JARVIS, an advanced AI assistant.
</s>
<|user|>
Hello JARVIS, what can you help me with today?
</s>
<|assistant|>
"""

    print(f"System: You are JARVIS, an advanced AI assistant.")
    print(f"User: Hello JARVIS, what can you help me with today?")

    start = time.time()
    output = llm(
        chat_prompt,
        max_tokens=100,
        temperature=0.7,
        stop=["</s>", "<|user|>"],
        echo=False,
    )
    inference_time = time.time() - start

    response = output["choices"][0]["text"].strip()
    tokens_generated = output["usage"]["completion_tokens"]

    print(f"\nJARVIS: {response}")
    print(f"\nTime: {inference_time:.2f}s ({tokens_generated / inference_time:.1f} tokens/sec)")

    # Performance summary
    print("\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)
    print(f"Model load time: {load_time:.2f}s")
    print(f"Inference speed: ~{tokens_generated / inference_time:.1f} tokens/sec")
    print(f"Memory: ~{model_size_mb * 1.2:.0f} MB estimated")
    print("\nJARVIS-Prime local inference is working!")

    return True


def test_model_info():
    """Print model metadata."""
    try:
        from llama_cpp import Llama

        llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=512,
            verbose=False,
        )

        print("\n" + "=" * 60)
        print("Model Info")
        print("=" * 60)

        # Get model metadata if available
        if hasattr(llm, 'metadata'):
            for key, value in llm.metadata.items():
                print(f"{key}: {value}")
        else:
            print(f"Path: {MODEL_PATH}")
            print(f"Context size: 2048")
            print(f"Quantization: Q4_K_M (4-bit)")

    except Exception as e:
        print(f"Could not get model info: {e}")


if __name__ == "__main__":
    try:
        test_basic_inference()
        test_model_info()
    except KeyboardInterrupt:
        print("\nTest interrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
