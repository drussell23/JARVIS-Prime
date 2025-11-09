#!/usr/bin/env python3
"""
Test script to verify all JARVIS Prime imports work correctly
"""
import sys

def test_basic_import():
    """Test basic package import"""
    print("Testing basic import...")
    import jarvis_prime
    print(f"✅ jarvis_prime v{jarvis_prime.__version__}")

def test_config_import():
    """Test configuration imports"""
    print("\nTesting config imports...")
    from jarvis_prime.configs import (
        LlamaModelConfig,
        LlamaPresets,
        QuantizationConfig,
        LoRAConfig,
        TrainingConfig,
    )
    print("✅ All config classes imported")

    # Test preset
    config = LlamaPresets.llama_13b_gcp_training()
    print(f"✅ Created config: {config.model_name}")

def test_model_import():
    """Test model imports (lazy)"""
    print("\nTesting model imports...")
    from jarvis_prime import LlamaModel, LlamaPresets
    print("✅ LlamaModel class imported")

    # Create model instance (don't load weights)
    config = LlamaPresets.llama_13b_m1_inference()
    config.model_name = "meta-llama/Llama-2-7b-hf"  # Use smaller model for testing
    model = LlamaModel(config)
    print(f"✅ Created LlamaModel instance: {model.device}")

def test_core_dependencies():
    """Test core dependencies are available"""
    print("\nTesting core dependencies...")
    import torch
    import transformers
    import peft
    import accelerate

    print(f"✅ torch {torch.__version__}")
    print(f"✅ transformers {transformers.__version__}")
    print(f"✅ peft {peft.__version__}")
    print(f"✅ accelerate {accelerate.__version__}")

def main():
    """Run all tests"""
    print("=" * 60)
    print("JARVIS Prime Import Tests")
    print("=" * 60)

    try:
        test_basic_import()
        test_config_import()
        test_core_dependencies()
        test_model_import()

        print("\n" + "=" * 60)
        print("🎉 All tests passed!")
        print("=" * 60)
        return 0

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
