"""
JARVIS Prime - Specialized PRIME Models for JARVIS
"""

__version__ = "0.6.0"

# Lazy imports to avoid loading heavy dependencies at import time
def __getattr__(name):
    if name == "PrimeModel":
        from jarvis_prime.models import PrimeModel
        return PrimeModel
    elif name == "PrimeTrainer":
        from jarvis_prime.trainer import PrimeTrainer
        return PrimeTrainer
    elif name == "LlamaModel":
        from jarvis_prime.models import LlamaModel
        return LlamaModel
    elif name == "LlamaTrainer":
        from jarvis_prime.trainer import LlamaTrainer
        return LlamaTrainer
    elif name == "LlamaPresets":
        from jarvis_prime.configs import LlamaPresets
        return LlamaPresets
    elif name == "LlamaModelConfig":
        from jarvis_prime.configs import LlamaModelConfig
        return LlamaModelConfig
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "PrimeModel",
    "PrimeTrainer",
    "LlamaModel",
    "LlamaTrainer",
    "LlamaPresets",
    "LlamaModelConfig",
    "__version__",
]
