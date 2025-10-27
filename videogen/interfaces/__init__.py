"""
VideoGen Interfaces Package
Web UI, CLI, and API interfaces
"""

from .gradio_ui import VideoGenUI
from .cli import VideoGenCLI

__all__ = ["VideoGenUI", "VideoGenCLI"]