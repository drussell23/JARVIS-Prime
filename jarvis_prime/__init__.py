"""
JARVIS Prime - Specialized PRIME Models for JARVIS
"""

__version__ = "0.6.0"

from jarvis_prime.models import PrimeModel
from jarvis_prime.trainer import PrimeTrainer

__all__ = [
    "PrimeModel",
    "PrimeTrainer",
    "__version__",
]
