"""
Base Generator Class - Core functionality for all video generation types
"""

import os
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod
from PIL import Image
import cv2
from pathlib import Path

from ..utils.config import Config
from ..utils.memory import MemoryManager
from ..utils.presets import PresetManager


class BaseGenerator(ABC):
    """
    Abstract base class for all video generators.
    Provides common functionality for cloud GPU optimization and memory management.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the base generator with configuration."""
        self.config = Config(config)
        self.device = self._setup_device()
        self.memory_manager = MemoryManager(self.device)
        self.preset_manager = PresetManager()
        
        # Performance settings
        self.fp16 = self._check_fp16_support()
        self.memory_efficient = True
        
        # Generation parameters
        self.default_params = {
            "num_frames": 16,
            "fps": 8,
            "guidance_scale": 7.5,
            "inference_steps": 20,
            "resolution": (512, 512),
            "seed": None
        }
        
        # Cloud GPU optimizations
        self.setup_cloud_optimizations()
        
    def _setup_device(self) -> torch.device:
        """Setup compute device with cloud GPU optimization."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Print GPU info for cloud environments
            gpu_name = torch.cuda.get_device_name(0)
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"Using GPU: {gpu_name}")
            print(f"GPU Memory: {memory_total:.1f}GB")
        else:
            device = torch.device("cpu")
            print("Warning: No GPU detected, using CPU (will be slow)")
        return device
    
    def _check_fp16_support(self) -> bool:
        """Check if GPU supports fp16 for memory optimization."""
        if not torch.cuda.is_available():
            return False
        try:
            # Test if the GPU supports fp16
            with torch.cuda.device(0):
                torch.tensor([1.0], dtype=torch.float16, device='cuda')
                return True
        except:
            return False
    
    def setup_cloud_optimizations(self):
        """Apply cloud-specific optimizations."""
        if self.device.type == 'cuda':
            # Enable memory-efficient attention if available
            torch.backends.cuda.enable_flash_sdp(True)
            
            # Set memory allocation strategy
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            
            # Enable benchmark mode for better performance
            torch.backends.cudnn.benchmark = True
            
            print("Cloud GPU optimizations enabled")
    
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """Preprocess image for model input."""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Resize to target resolution
        image = image.resize(self.config.resolution, Image.LANCZOS)
        
        # Convert to tensor and normalize
        image_tensor = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_tensor).permute(2, 0, 1)
        
        return image_tensor.to(self.device)
    
    def postprocess_video(self, frames: torch.Tensor, fps: int = 8) -> np.ndarray:
        """Convert tensor frames to video array."""
        # Move to CPU and convert to numpy
        frames = frames.detach().cpu()
        
        # Convert from (B, C, H, W) to (B, H, W, C)
        if len(frames.shape) == 4:
            frames = frames.permute(0, 2, 3, 1)
        
        # Convert to numpy and ensure values are in [0, 1]
        frames = frames.numpy()
        frames = np.clip(frames, 0, 1)
        
        # Convert to uint8
        frames = (frames * 255).astype(np.uint8)
        
        return frames
    
    def save_video(self, frames: np.ndarray, output_path: str, fps: int = 8) -> str:
        """Save frames as video file."""
        if not output_path.endswith(('.mp4', '.avi', '.mov')):
            output_path += '.mp4'
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Ensure frames are in correct format
        if len(frames.shape) == 4:  # (N, H, W, 3)
            height, width = frames.shape[1], frames.shape[2]
        else:
            raise ValueError("Invalid frame format")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Write frames
        for frame in frames:
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        video_writer.release()
        print(f"Video saved to: {output_path}")
        
        return output_path
    
    def monitor_memory(self) -> Dict[str, float]:
        """Monitor GPU memory usage."""
        return self.memory_manager.get_memory_info()
    
    def cleanup_memory(self):
        """Clean up GPU memory."""
        torch.cuda.empty_cache()
    
    @abstractmethod
    def generate(self, *args, **kwargs) -> str:
        """Abstract method for generating videos. Must be implemented by subclasses."""
        pass
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get device and performance information."""
        info = {
            "device": str(self.device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0,
            "memory_available": torch.cuda.get_device_properties(0).reserved_memory / 1024**3 if torch.cuda.is_available() else 0,
            "fp16_supported": self.fp16,
            "memory_efficient": self.memory_efficient
        }
        return info
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_memory()