"""
VideoGen - Cloud-Optimized Video Generation System
Core package with generators and utilities
"""

from .generators.base_generator import BaseGenerator
from .generators.cinematic_generator import CinematicGenerator
from .generators.product_generator import ProductGenerator
from .generators.explainer_generator import ExplainerGenerator
from .utils.config import Config
from .utils.memory import MemoryManager
from .utils.presets import PresetManager

__version__ = "1.0.0"
__all__ = [
    "BaseGenerator",
    "CinematicGenerator", 
    "ProductGenerator",
    "ExplainerGenerator",
    "Config",
    "MemoryManager",
    "PresetManager"
]