#!/usr/bin/env python3
"""
Example: Run inference with Llama-2-13B on M1 Mac

This example demonstrates:
- Loading quantized Llama-2-13B on M1 Mac (16GB)
- Synchronous generation
- Async generation with batching
- Chat interface
- Memory-efficient inference
"""
import asyncio
import logging

from jarvis_prime.configs.llama_config import LlamaPresets
from jarvis_prime.models.llama_model import LlamaModel, load_llama_13b_m1

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_generation():
    """Basic text generation"""
    logger.info("=== Basic Generation ===")

    # Load model optimized for M1 Mac
    model = load_llama_13b_m1()

    # Generate
    prompt = "What is the meaning of artificial intelligence?"
    response = model.generate(prompt, max_length=512, temperature=0.7)

    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")

    # Memory usage
    memory = model.get_memory_usage()
    print(f"\nMemory usage: {memory}")


def example_batch_generation():
    """Batch generation"""
    logger.info("\n=== Batch Generation ===")

    model = load_llama_13b_m1()

    prompts = [
        "Explain quantum computing in simple terms.",
        "What are the benefits of machine learning?",
        "How does a neural network work?",
    ]

    responses = model.generate(prompts, max_length=256, temperature=0.7)

    for prompt, response in zip(prompts, responses):
        print(f"\n📝 Prompt: {prompt}")
        print(f"💬 Response: {response}")


def example_chat_interface():
    """Chat interface with conversation history"""
    logger.info("\n=== Chat Interface ===")

    model = load_llama_13b_m1()

    # Conversation
    messages = [
        {"role": "system", "content": "You are JARVIS, an advanced AI assistant created to help users."},
        {"role": "user", "content": "Hello! What can you help me with?"},
    ]

    response = model.chat(messages)
    print(f"\n🤖 JARVIS: {response}")

    # Continue conversation
    messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": "Can you help me understand reinforcement learning?"})

    response = model.chat(messages)
    print(f"\n🤖 JARVIS: {response}")


async def example_async_generation():
    """Async generation with concurrent requests"""
    logger.info("\n=== Async Generation ===")

    # Load model with async enabled
    config = LlamaPresets.llama_13b_m1_inference()
    config.inference.async_enabled = True
    config.inference.max_concurrent_requests = 50

    model = LlamaModel(config)
    model.load()

    # Create multiple concurrent requests
    prompts = [
        "What is machine learning?",
        "Explain deep learning.",
        "What is reinforcement learning?",
        "How does transfer learning work?",
        "What are transformers in AI?",
    ]

    # Run concurrently
    tasks = [
        model.generate_async(prompt, max_length=200)
        for prompt in prompts
    ]

    responses = await asyncio.gather(*tasks)

    for prompt, response in zip(prompts, responses):
        print(f"\n📝 {prompt}")
        print(f"💬 {response}")


async def example_async_chat():
    """Async chat interface"""
    logger.info("\n=== Async Chat ===")

    config = LlamaPresets.llama_13b_m1_inference()
    config.inference.async_enabled = True

    model = LlamaModel(config)
    model.load()

    messages = [
        {"role": "system", "content": "You are JARVIS, a helpful AI assistant."},
        {"role": "user", "content": "What's the weather like on Mars?"},
    ]

    response = await model.chat_async(messages)
    print(f"\n🤖 JARVIS: {response}")


def example_custom_config():
    """Using custom configuration"""
    logger.info("\n=== Custom Configuration ===")

    # Create custom config
    config = LlamaPresets.llama_13b_m1_inference()

    # Customize inference settings
    config.inference.temperature = 0.9
    config.inference.top_p = 0.95
    config.inference.max_length = 1024

    # Customize quantization
    config.quantization.bits = 4  # More aggressive quantization

    model = LlamaModel(config)
    model.load()

    prompt = "Tell me a creative story about AI and humanity."
    response = model.generate(prompt)

    print(f"\n📝 Prompt: {prompt}")
    print(f"💬 Response: {response}")


def example_load_adapter():
    """Load fine-tuned LoRA adapter"""
    logger.info("\n=== Load Fine-tuned Adapter ===")

    # Load base model
    model = load_llama_13b_m1()

    # Load your trained adapter
    adapter_path = "./outputs/llama-13b-jarvis"
    model.load_adapter(adapter_path)

    # Now use fine-tuned model
    prompt = "Hello JARVIS, what's your purpose?"
    response = model.generate(prompt)

    print(f"\n📝 Prompt: {prompt}")
    print(f"💬 Response: {response}")


def main():
    """Run all examples"""

    # Synchronous examples
    example_basic_generation()
    example_batch_generation()
    example_chat_interface()
    example_custom_config()

    # Uncomment to test adapter loading
    # example_load_adapter()

    # Async examples
    asyncio.run(example_async_generation())
    asyncio.run(example_async_chat())


if __name__ == "__main__":
    main()
