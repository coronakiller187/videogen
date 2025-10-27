"""
Memory Management for Cloud GPU Environments
Optimizes memory usage and prevents OOM errors in cloud setups
"""

import torch
import gc
import psutil
import os
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class MemoryInfo:
    """Memory usage information."""
    total_gb: float
    used_gb: float
    free_gb: float
    percent_used: float
    gpu_memory_gb: Optional[float] = None
    gpu_used_gb: Optional[float] = None
    gpu_percent_used: Optional[float] = None


class MemoryManager:
    """
    Memory manager for cloud GPU environments.
    Handles memory optimization, monitoring, and cleanup.
    """
    
    def __init__(self, device: torch.device, max_memory_fraction: float = 0.9):
        """
        Initialize memory manager.
        
        Args:
            device: PyTorch device
            max_memory_fraction: Maximum memory usage fraction (0.0-1.0)
        """
        self.device = device
        self.max_memory_fraction = max_memory_fraction
        self.monitoring_enabled = True
        self.cleanup_threshold = 0.85  # 85% memory usage threshold
        
        # Set GPU memory management
        if device.type == 'cuda':
            self._setup_gpu_memory()
        
        print(f"💾 Memory Manager initialized for {device}")
    
    def _setup_gpu_memory(self):
        """Setup GPU memory management."""
        try:
            # Enable memory growth
            torch.cuda.empty_cache()
            
            # Set memory fraction
            total_memory = torch.cuda.get_device_properties(0).total_memory
            max_memory = int(total_memory * self.max_memory_fraction)
            
            # This would be set on the model, not globally
            print(f"🖥️ GPU Memory: {total_memory / 1024**3:.1f}GB total, {max_memory / 1024**3:.1f}GB max")
            
        except Exception as e:
            print(f"⚠️ GPU memory setup warning: {e}")
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get comprehensive memory information."""
        info = {}
        
        # System memory
        memory = psutil.virtual_memory()
        info.update({
            "system_total_gb": memory.total / 1024**3,
            "system_used_gb": memory.used / 1024**3,
            "system_free_gb": memory.available / 1024**3,
            "system_percent_used": memory.percent
        })
        
        # GPU memory
        if self.device.type == 'cuda':
            try:
                gpu_memory = torch.cuda.get_device_properties(0)
                gpu_allocated = torch.cuda.memory_allocated(0)
                gpu_reserved = torch.cuda.memory_reserved(0)
                
                info.update({
                    "gpu_total_gb": gpu_memory.total_memory / 1024**3,
                    "gpu_allocated_gb": gpu_allocated / 1024**3,
                    "gpu_reserved_gb": gpu_reserved / 1024**3,
                    "gpu_used_gb": gpu_reserved / 1024**3,  # Reserved = used
                    "gpu_percent_used": (gpu_reserved / gpu_memory.total_memory) * 100,
                    "gpu_name": gpu_memory.name
                })
            except Exception as e:
                info.update({
                    "gpu_error": str(e)
                })
        
        return info
    
    def get_memory_info_object(self) -> MemoryInfo:
        """Get memory info as an object."""
        info_dict = self.get_memory_info()
        
        return MemoryInfo(
            total_gb=info_dict.get("system_total_gb", 0),
            used_gb=info_dict.get("system_used_gb", 0),
            free_gb=info_dict.get("system_free_gb", 0),
            percent_used=info_dict.get("system_percent_used", 0),
            gpu_memory_gb=info_dict.get("gpu_total_gb"),
            gpu_used_gb=info_dict.get("gpu_used_gb"),
            gpu_percent_used=info_dict.get("gpu_percent_used")
        )
    
    def check_memory_pressure(self, threshold: float = 0.8) -> bool:
        """
        Check if memory pressure is high.
        
        Args:
            threshold: Memory usage threshold (0.0-1.0)
            
        Returns:
            True if memory pressure is high
        """
        memory_info = self.get_memory_info()
        
        # Check system memory
        system_pressure = memory_info["system_percent_used"] / 100 > threshold
        
        # Check GPU memory if available
        gpu_pressure = False
        if memory_info.get("gpu_percent_used"):
            gpu_pressure = memory_info["gpu_percent_used"] / 100 > threshold
        
        return system_pressure or gpu_pressure
    
    def monitor_memory_during_generation(self, operation_name: str = "operation"):
        """
        Context manager for monitoring memory during operations.
        
        Args:
            operation_name: Name of the operation being monitored
            
        Usage:
            with memory_manager.monitor_memory_during_generation("video_generation"):
                # Your generation code here
                pass
        """
        return MemoryMonitorContext(self, operation_name)
    
    def cleanup_memory(self):
        """Perform comprehensive memory cleanup."""
        # System garbage collection
        gc.collect()
        
        # GPU memory cleanup
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
    
    def cleanup_memory_aggressive(self):
        """Perform aggressive memory cleanup."""
        print("🧹 Performing aggressive memory cleanup...")
        
        # Multiple garbage collection passes
        for _ in range(3):
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # Small delay to allow cleanup
        time.sleep(0.5)
        
        final_info = self.get_memory_info()
        print(f"✅ Memory after cleanup - GPU: {final_info.get('gpu_percent_used', 0):.1f}%")
    
    def optimize_for_generation(self, num_frames: int, resolution: tuple) -> Dict[str, Any]:
        """
        Calculate optimal settings for generation based on available memory.
        
        Args:
            num_frames: Number of frames to generate
            resolution: Video resolution (width, height)
            
        Returns:
            Dictionary with optimized settings
        """
        memory_info = self.get_memory_info()
        
        # Estimate memory requirements
        pixels_per_frame = resolution[0] * resolution[1]
        total_pixels = pixels_per_frame * num_frames
        
        # Rough estimation: 1GB per 100M pixels (adjust as needed)
        estimated_memory_gb = total_pixels / (100 * 1024 * 1024)  # 100M pixels per GB
        
        available_memory_gb = memory_info.get("gpu_used_gb", 0)
        if not available_memory_gb:
            available_memory_gb = memory_info.get("system_free_gb", 0) * 0.5  # Conservative estimate
        
        # Calculate optimal settings
        if estimated_memory_gb > available_memory_gb * 0.8:
            print(f"⚠️ High memory usage detected. Estimated: {estimated_memory_gb:.1f}GB, Available: {available_memory_gb:.1f}GB")
            
            # Reduce resolution and frames
            scale_factor = (available_memory_gb * 0.7) / estimated_memory_gb
            new_resolution = (int(resolution[0] * scale_factor), int(resolution[1] * scale_factor))
            
            return {
                "resolution": new_resolution,
                "num_frames": max(num_frames // 2, 8),
                "batch_size": 1,
                "fp16": True,
                "memory_efficient": True,
                "warning": "Memory-optimized settings applied"
            }
        else:
            return {
                "resolution": resolution,
                "num_frames": num_frames,
                "batch_size": 2 if available_memory_gb > 12 else 1,
                "fp16": True,
                "memory_efficient": True
            }
    
    def batch_generate_with_memory_management(self, generate_func, params_list: list) -> list:
        """
        Generate videos in batches with memory management.
        
        Args:
            generate_func: Function to call for generation
            params_list: List of parameter dictionaries
            
        Returns:
            List of generation results
        """
        results = []
        batch_size = self._get_optimal_batch_size()
        
        for i in range(0, len(params_list), batch_size):
            batch = params_list[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(params_list)-1)//batch_size + 1} ({len(batch)} items)")
            
            # Monitor memory before batch
            memory_before = self.get_memory_info()
            
            # Generate batch
            batch_results = []
            for j, params in enumerate(batch):
                try:
                    result = generate_func(**params)
                    batch_results.append(result)
                    
                    # Cleanup after each generation if memory is high
                    if memory_before.get("gpu_percent_used", 0) > 80:
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"❌ Generation failed for batch {i//batch_size + 1}, item {j}: {e}")
                    batch_results.append(None)
            
            results.extend(batch_results)
            
            # Memory cleanup between batches
            if i + batch_size < len(params_list):
                self.cleanup_memory()
                memory_after = self.get_memory_info()
                print(f"📊 Memory after batch: {memory_after.get('gpu_percent_used', 0):.1f}%")
        
        return results
    
    def _get_optimal_batch_size(self) -> int:
        """Get optimal batch size based on available memory."""
        memory_info = self.get_memory_info()
        gpu_memory_gb = memory_info.get("gpu_total_gb", 0)
        
        if gpu_memory_gb >= 24:  # A100, H100
            return 3
        elif gpu_memory_gb >= 16:  # RTX 4080, A6000
            return 2
        else:  # T4, RTX 3080, etc.
            return 1
    
    def detect_memory_leak(self, initial_memory: Optional[float] = None, duration_minutes: int = 10) -> Dict[str, Any]:
        """
        Detect potential memory leaks during long operations.
        
        Args:
            initial_memory: Initial memory usage (None for auto-detect)
            duration_minutes: Duration to monitor in minutes
            
        Returns:
            Memory leak analysis results
        """
        if initial_memory is None:
            initial_memory = self.get_memory_info().get("gpu_used_gb", 0)
        
        print(f"🔍 Monitoring memory for {duration_minutes} minutes to detect leaks...")
        
        start_time = time.time()
        memory_samples = []
        
        # Sample memory every 30 seconds
        while time.time() - start_time < duration_minutes * 60:
            current_memory = self.get_memory_info().get("gpu_used_gb", 0)
            memory_samples.append(current_memory)
            
            if len(memory_samples) > 1:
                memory_change = current_memory - memory_samples[-2]
                if abs(memory_change) > 0.1:  # 100MB threshold
                    print(f"📈 Memory change detected: {memory_change:.2f}GB")
            
            time.sleep(30)
        
        # Analyze results
        if len(memory_samples) > 2:
            final_memory = memory_samples[-1]
            memory_increase = final_memory - initial_memory
            
            leak_detected = memory_increase > 1.0  # 1GB threshold
            memory_growth_rate = memory_increase / (len(memory_samples) - 1)
            
            return {
                "leak_detected": leak_detected,
                "initial_memory_gb": initial_memory,
                "final_memory_gb": final_memory,
                "total_increase_gb": memory_increase,
                "growth_rate_gb_per_sample": memory_growth_rate,
                "samples_count": len(memory_samples),
                "recommendation": "Consider more frequent cleanup" if leak_detected else "Memory usage normal"
            }
        
        return {"error": "Not enough samples for analysis"}


class MemoryMonitorContext:
    """Context manager for monitoring memory during operations."""
    
    def __init__(self, memory_manager: MemoryManager, operation_name: str):
        self.memory_manager = memory_manager
        self.operation_name = operation_name
        self.memory_before = None
        
    def __enter__(self):
        self.memory_before = self.memory_manager.get_memory_info()
        print(f"🎬 Starting {self.operation_name}")
        print(f"📊 Memory before: GPU {self.memory_before.get('gpu_percent_used', 0):.1f}%")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        memory_after = self.memory_manager.get_memory_info()
        print(f"📊 Memory after: GPU {memory_after.get('gpu_percent_used', 0):.1f}%")
        
        # Memory change
        if self.memory_before and memory_after.get("gpu_percent_used"):
            change = memory_after["gpu_percent_used"] - self.memory_before["gpu_percent_used"]
            print(f"📈 Memory change: {change:+.1f}%")
        
        if exc_type:
            print(f"❌ {self.operation_name} failed: {exc_val}")
        else:
            print(f"✅ {self.operation_name} completed")
        
        # Cleanup if needed
        if memory_after.get("gpu_percent_used", 0) > 85:
            self.memory_manager.cleanup_memory()