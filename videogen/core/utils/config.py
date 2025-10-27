"""
Configuration management for VideoGen
Handles cloud GPU optimization and user preferences
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path


class Config:
    """Configuration manager for VideoGen with cloud GPU optimizations."""
    
    def __init__(self, config_dict: Optional[Dict] = None):
        """Initialize configuration with defaults and user settings."""
        
        # Default configuration
        self.default_config = {
            # Core settings
            "resolution": (512, 512),
            "fps": 8,
            "duration": 5.0,
            "guidance_scale": 7.5,
            "inference_steps": 20,
            "seed": None,
            
            # Cloud GPU optimizations
            "memory_efficient": True,
            "fp16": True,
            "batch_size": 1,
            "gradient_checkpointing": True,
            "max_memory_fraction": 0.9,
            
            # Quality settings
            "quality_mode": "balanced",  # fast, balanced, quality
            "noise_reduction": True,
            "motion_smoothing": True,
            
            # Output settings
            "output_format": "mp4",
            "output_codec": "h264",
            "compress_output": True,
            
            # Cloud environment detection
            "environment": self._detect_environment(),
            
            # GPU-specific settings
            "gpu_memory_growth": True,
            "cudnn_benchmark": True,
            "allow_tf32": True,
            
            # Logging and monitoring
            "verbose": True,
            "save_metadata": True,
            "monitor_memory": True,
            "memory_threshold": 0.85  # 85% memory usage threshold
        }
        
        # Load configuration
        if config_dict:
            self.config = self._merge_config(self.default_config, config_dict)
        else:
            self.config = self.default_config.copy()
        
        # Load from file if exists
        self.load_from_file()
        
        # Apply environment-specific optimizations
        self._apply_environment_optimizations()
    
    def _detect_environment(self) -> str:
        """Detect the current cloud environment."""
        if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
            return "colab"
        elif "KAGGLE_KERNEL_RUN_ENV" in os.environ:
            return "kaggle"
        elif "AWS_LAMBDA_FUNCTION_NAME" in os.environ:
            return "lambda"
        elif "GOOGLE_CLOUD_PROJECT" in os.environ:
            return "gcp"
        elif "AZURE_CLIENT_ID" in os.environ:
            return "azure"
        else:
            return "local"
    
    def _apply_environment_optimizations(self):
        """Apply optimizations based on detected environment."""
        env = self.config["environment"]
        
        if env == "colab":
            # Google Colab optimizations
            self.config.update({
                "memory_efficient": True,
                "fp16": True,
                "max_memory_fraction": 0.8,  # Conservative for Colab
                "batch_size": 1,
                "verbose": True
            })
        
        elif env == "kaggle":
            # Kaggle optimizations
            self.config.update({
                "memory_efficient": True,
                "fp16": True,
                "max_memory_fraction": 0.85,
                "batch_size": 1
            })
        
        elif env in ["gcp", "azure"]:
            # Cloud platform optimizations
            self.config.update({
                "memory_efficient": True,
                "fp16": True,
                "max_memory_fraction": 0.9,
                "batch_size": 2 if self.get_gpu_memory_gb() > 20 else 1
            })
    
    def _merge_config(self, default: Dict, user: Dict) -> Dict:
        """Merge user configuration with defaults."""
        merged = default.copy()
        merged.update(user)
        return merged
    
    def load_from_file(self, config_path: Optional[str] = None):
        """Load configuration from file."""
        if not config_path:
            config_path = self._get_default_config_path()
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    if config_path.endswith('.json'):
                        file_config = json.load(f)
                    elif config_path.endswith(('.yaml', '.yml')):
                        file_config = yaml.safe_load(f)
                    else:
                        raise ValueError(f"Unsupported config file format: {config_path}")
                
                self.config.update(file_config)
                print(f"✅ Loaded configuration from {config_path}")
                
            except Exception as e:
                print(f"⚠️ Failed to load config from {config_path}: {e}")
    
    def save_to_file(self, config_path: Optional[str] = None):
        """Save current configuration to file."""
        if not config_path:
            config_path = self._get_default_config_path()
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                if config_path.endswith('.json'):
                    json.dump(self.config, f, indent=2)
                elif config_path.endswith(('.yaml', '.yml')):
                    yaml.dump(self.config, f, default_flow_style=False)
                
            print(f"✅ Saved configuration to {config_path}")
            
        except Exception as e:
            print(f"❌ Failed to save config to {config_path}: {e}")
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        home_dir = Path.home()
        config_dir = home_dir / ".videogen"
        return str(config_dir / "config.yaml")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value
    
    def update(self, updates: Dict[str, Any]):
        """Update multiple configuration values."""
        self.config.update(updates)
    
    def get_resolution(self) -> tuple:
        """Get video resolution."""
        return self.config["resolution"]
    
    def set_resolution(self, width: int, height: int):
        """Set video resolution."""
        self.config["resolution"] = (width, height)
    
    def get_gpu_memory_gb(self) -> float:
        """Get available GPU memory in GB."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / 1024**3
        except:
            pass
        return 0.0
    
    def get_memory_optimized_settings(self) -> Dict[str, Any]:
        """Get memory-optimized settings based on current GPU."""
        memory_gb = self.get_gpu_memory_gb()
        
        if memory_gb >= 24:  # A100, H100
            return {
                "resolution": (1024, 1024),
                "batch_size": 2,
                "inference_steps": 25,
                "quality_mode": "quality"
            }
        elif memory_gb >= 12:  # RTX 3090, 4090
            return {
                "resolution": (768, 768),
                "batch_size": 2,
                "inference_steps": 22,
                "quality_mode": "balanced"
            }
        elif memory_gb >= 8:   # RTX 3080, T4
            return {
                "resolution": (512, 512),
                "batch_size": 1,
                "inference_steps": 18,
                "quality_mode": "balanced"
            }
        else:  # Limited memory
            return {
                "resolution": (384, 384),
                "batch_size": 1,
                "inference_steps": 15,
                "quality_mode": "fast"
            }
    
    def apply_quality_preset(self, preset: str):
        """Apply quality preset settings."""
        presets = {
            "fast": {
                "resolution": (384, 384),
                "inference_steps": 15,
                "guidance_scale": 7.0,
                "batch_size": 1
            },
            "balanced": {
                "resolution": (512, 512),
                "inference_steps": 20,
                "guidance_scale": 7.5,
                "batch_size": 1
            },
            "quality": {
                "resolution": (768, 768),
                "inference_steps": 25,
                "guidance_scale": 8.0,
                "batch_size": 1
            },
            "ultra": {
                "resolution": (1024, 1024),
                "inference_steps": 30,
                "guidance_scale": 8.5,
                "batch_size": 1
            }
        }
        
        if preset in presets:
            self.update(presets[preset])
            print(f"Applied quality preset: {preset}")
        else:
            print(f"Unknown quality preset: {preset}. Available: {list(presets.keys())}")
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get information about the current environment."""
        env = self.config["environment"]
        
        info = {
            "environment": env,
            "gpu_memory_gb": self.get_gpu_memory_gb(),
            "optimal_settings": self.get_memory_optimized_settings()
        }
        
        # Add environment-specific info
        if env == "colab":
            info.update({
                "colab_runtime": os.environ.get("COLAB_GPU", "CPU"),
                "colab_session": "Notebook" if "IPython" in os.environ else "Unknown"
            })
        elif env == "kaggle":
            info.update({
                "kaggle_gpu": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "CPU")
            })
        
        return info
    
    def is_cloud_environment(self) -> bool:
        """Check if running in a cloud environment."""
        return self.config["environment"] != "local"
    
    def get_cloud_specific_settings(self) -> Dict[str, Any]:
        """Get cloud-specific optimization settings."""
        env = self.config["environment"]
        
        if env == "colab":
            return {
                "memory_efficient": True,
                "fp16": True,
                "clean_memory_after_gen": True,
                "monitor_usage": True
            }
        elif env == "kaggle":
            return {
                "memory_efficient": True,
                "fp16": True,
                "limit_concurrent_generations": True
            }
        else:
            return {
                "memory_efficient": True,
                "fp16": True
            }
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.config[key]
    
    def __setitem__(self, key: str, value: Any):
        """Allow dictionary-style assignment."""
        self.config[key] = value
    
    def __contains__(self, key: str) -> bool:
        """Allow 'in' operator."""
        return key in self.config
    
    def keys(self):
        """Return configuration keys."""
        return self.config.keys()
    
    def values(self):
        """Return configuration values."""
        return self.config.values()
    
    def items(self):
        """Return configuration items."""
        return self.config.items()