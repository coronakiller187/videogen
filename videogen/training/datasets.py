"""
Dataset Management for VideoGen
Loading, preprocessing, and managing training datasets
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import numpy as np
from PIL import Image


class DatasetBuilder:
    """Build and manage training datasets."""
    
    def __init__(self, dataset_name: str = "custom_dataset"):
        """Initialize dataset builder."""
        self.dataset_name = dataset_name
        self.image_paths = []
        self.text_prompts = []
        self.metadata = {}
    
    def add_video_samples(self, 
                         video_paths: List[str],
                         prompts: List[str],
                         metadata: Optional[Dict] = None):
        """Add video samples to dataset."""
        if len(video_paths) != len(prompts):
            raise ValueError("Number of video paths must match number of prompts")
        
        self.image_paths.extend(video_paths)
        self.text_prompts.extend(prompts)
        
        if metadata:
            self.metadata.update(metadata)
        
        print(f"✅ Added {len(video_paths)} video samples to dataset")
    
    def add_image_samples(self,
                         image_paths: List[str],
                         prompts: List[str],
                         animation_prompts: Optional[List[str]] = None):
        """Add image samples for image-to-video training."""
        if len(image_paths) != len(prompts):
            raise ValueError("Number of image paths must match number of prompts")
        
        self.image_paths.extend(image_paths)
        self.text_prompts.extend(prompts)
        
        if animation_prompts:
            self.text_prompts.extend(animation_prompts)
        
        print(f"✅ Added {len(image_paths)} image samples to dataset")
    
    def save_dataset(self, output_dir: str):
        """Save dataset to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata_file = output_path / "metadata.json"
        metadata = {
            "name": self.dataset_name,
            "samples": len(self.image_paths),
            "has_images": len(self.image_paths) > 0,
            "has_prompts": len(self.text_prompts) > 0,
            "custom_metadata": self.metadata
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save sample list
        samples_file = output_path / "samples.json"
        samples = []
        for i, (img_path, prompt) in enumerate(zip(self.image_paths, self.text_prompts)):
            samples.append({
                "id": i,
                "image_path": str(img_path),
                "prompt": prompt,
                "type": "video" if img_path.endswith(('.mp4', '.avi', '.mov')) else "image"
            })
        
        with open(samples_file, 'w') as f:
            json.dump(samples, f, indent=2)
        
        print(f"✅ Dataset saved to {output_path}")
        return str(output_path)


class DataLoader:
    """Data loader for training video generation models."""
    
    def __init__(self, dataset_path: str, batch_size: int = 1):
        """Initialize data loader."""
        self.dataset_path = Path(dataset_path)
        self.batch_size = batch_size
        
        # Load dataset
        self.samples = self._load_dataset()
        print(f"📚 Loaded dataset with {len(self.samples)} samples")
    
    def _load_dataset(self) -> List[Dict]:
        """Load dataset from saved files."""
        samples_file = self.dataset_path / "samples.json"
        
        if not samples_file.exists():
            raise FileNotFoundError(f"Dataset samples file not found: {samples_file}")
        
        with open(samples_file, 'r') as f:
            samples = json.load(f)
        
        return samples
    
    def get_batch(self, index: int) -> Dict[str, Any]:
        """Get a batch of training data."""
        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_samples = self.samples[start_idx:end_idx]
        
        # Process batch (placeholder implementation)
        batch = {
            "images": [],  # Would load and preprocess images
            "prompts": [sample["prompt"] for sample in batch_samples],
            "sample_info": batch_samples
        }
        
        return batch
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.samples) // self.batch_size
    
    def __iter__(self):
        """Iterate over batches."""
        for i in range(len(self)):
            yield self.get_batch(i)