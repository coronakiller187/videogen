"""
Cinematic Generator - Specialized for cinematic storytelling
Handles camera movements, lighting, and cinematic effects
"""

import os
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
import json

from .base_generator import BaseGenerator
from ..utils.config import Config


class CinematicGenerator(BaseGenerator):
    """
    Cinematic video generator with professional camera movements and lighting control.
    Designed for cinematic storytelling with movie-quality output.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize cinematic generator."""
        super().__init__(config)
        
        # Cinematic-specific defaults
        self.default_params.update({
            "cinematic_style": "hollywood",
            "camera_movement": "static",
            "lighting": "natural",
            "color_grading": "cinematic",
            "depth_of_field": True,
            "motion_blur": True
        })
        
        # Load cinematic presets
        self._load_cinematic_presets()
        
        # Camera and lighting systems
        self.camera_movements = self.preset_manager.get_presets("camera_movements")
        self.lighting_presets = self.preset_manager.get_presets("lighting")
        self.color_grades = self.preset_manager.get_presets("color_grading")
        
    def _load_cinematic_presets(self):
        """Load cinematic presets for camera movements and lighting."""
        # Camera movement presets
        self.camera_presets = {
            "static": {
                "description": "No camera movement, stable shots",
                "motion_params": {"dolly": 0.0, "pan": 0.0, "tilt": 0.0, "roll": 0.0}
            },
            "slow_dolly_in": {
                "description": "Slow push-in towards subject",
                "motion_params": {"dolly": 1.0, "pan": 0.0, "tilt": 0.0, "roll": 0.0},
                "easing": "smooth"
            },
            "dolly_out": {
                "description": "Pull away from subject",
                "motion_params": {"dolly": -1.0, "pan": 0.0, "tilt": 0.0, "roll": 0.0}
            },
            "tracking_left": {
                "description": "Camera tracks left following action",
                "motion_params": {"dolly": 0.0, "pan": -0.5, "tilt": 0.0, "roll": 0.0}
            },
            "tracking_right": {
                "description": "Camera tracks right following action", 
                "motion_params": {"dolly": 0.0, "pan": 0.5, "tilt": 0.0, "roll": 0.0}
            },
            "dutch_angle": {
                "description": "Tilted camera angle for tension",
                "motion_params": {"dolly": 0.0, "pan": 0.0, "tilt": 0.0, "roll": 15.0}
            },
            "crane_up": {
                "description": "Camera moves up revealing scene",
                "motion_params": {"dolly": 0.0, "pan": 0.0, "tilt": 1.0, "roll": 0.0}
            },
            "orbital": {
                "description": "Camera orbits around subject",
                "motion_params": {"dolly": 0.0, "pan": 1.0, "tilt": 0.2, "roll": 0.0}
            }
        }
        
        # Lighting presets
        self.lighting_presets = {
            "natural": {
                "description": "Natural daylight lighting",
                "lighting_params": {
                    "primary_light": "soft_directional",
                    "fill_light": "ambient",
                    "rim_light": None,
                    "color_temperature": 5600
                }
            },
            "golden_hour": {
                "description": "Warm sunset/sunrise lighting",
                "lighting_params": {
                    "primary_light": "directional_warm",
                    "fill_light": "warm_ambient", 
                    "rim_light": "warm_back",
                    "color_temperature": 3200
                }
            },
            "blue_hour": {
                "description": "Cool twilight lighting",
                "lighting_params": {
                    "primary_light": "directional_cool",
                    "fill_light": "cool_ambient",
                    "rim_light": "cool_back", 
                    "color_temperature": 8000
                }
            },
            "dramatic": {
                "description": "High contrast dramatic lighting",
                "lighting_params": {
                    "primary_light": "hard_directional",
                    "fill_light": "minimal",
                    "rim_light": "strong_back",
                    "color_temperature": 5600
                }
            },
            "neon": {
                "description": "Vibrant neon lighting for cyberpunk",
                "lighting_params": {
                    "primary_light": "neon_pink",
                    "fill_light": "neon_cyan",
                    "rim_light": "neon_purple",
                    "color_temperature": 0  # RGB values
                }
            },
            "moonlight": {
                "description": "Cool moonlight for night scenes",
                "lighting_params": {
                    "primary_light": "cool_directional",
                    "fill_light": "cool_ambient",
                    "rim_light": "cool_back",
                    "color_temperature": 12000
                }
            }
        }
        
        # Color grading presets
        self.color_presets = {
            "cinematic": {
                "description": "Professional cinematic color grade",
                "color_params": {
                    "contrast": 1.2,
                    "saturation": 0.9,
                    "shadows": -0.1,
                    "highlights": 0.1,
                    "vignette": 0.15
                }
            },
            "filmic": {
                "description": "Familiar film look",
                "color_params": {
                    "contrast": 1.1,
                    "saturation": 0.95,
                    "shadows": -0.05,
                    "highlights": 0.05,
                    "grain": 0.1
                }
            },
            "noir": {
                "description": "Classic black and white noir",
                "color_params": {
                    "contrast": 1.5,
                    "saturation": 0.0,
                    "shadows": -0.2,
                    "highlights": 0.2,
                    "vignette": 0.3
                }
            },
            "vibrant": {
                "description": "High saturation vibrant colors",
                "color_params": {
                    "contrast": 1.1,
                    "saturation": 1.3,
                    "shadows": 0.0,
                    "highlights": 0.1,
                    "vignette": 0.0
                }
            }
        }
    
    def _enhance_prompt_with_cinematic_elements(self, prompt: str, style_params: Dict) -> str:
        """Enhance prompt with cinematic elements based on style parameters."""
        enhanced_prompt = prompt
        
        # Add cinematic terms to prompt
        if style_params.get("cinematic_style") == "hollywood":
            enhanced_prompt += ", cinematic composition, professional cinematography"
        elif style_params.get("cinematic_style") == "indie":
            enhanced_prompt += ", indie film aesthetic, natural lighting"
        elif style_params.get("cinematic_style") == "arthouse":
            enhanced_prompt += ", experimental cinematography, artistic framing"
        
        # Add camera movement description
        camera_move = style_params.get("camera_movement", "static")
        if camera_move != "static":
            move_desc = self.camera_presets.get(camera_move, {}).get("description", "")
            if move_desc:
                enhanced_prompt += f", {move_desc}"
        
        # Add lighting description
        lighting = style_params.get("lighting", "natural")
        if lighting in self.lighting_presets:
            lighting_desc = self.lighting_presets[lighting].get("description", "")
            if lighting_desc:
                enhanced_prompt += f", {lighting_desc}"
        
        # Add technical specifications for quality
        enhanced_prompt += ", high production value, sharp focus, detailed textures"
        
        return enhanced_prompt
    
    def generate(self, 
                prompt: str,
                style: str = "cinematic",
                camera_movement: str = "static", 
                lighting: str = "natural",
                color_grading: str = "cinematic",
                duration: float = 5.0,
                fps: int = 8,
                resolution: Tuple[int, int] = (512, 512),
                seed: Optional[int] = None,
                output_path: str = "outputs/cinematic_video.mp4",
                **kwargs) -> str:
        """
        Generate cinematic video from prompt.
        
        Args:
            prompt: Text description for video content
            style: Cinematic style (hollywood, indie, arthouse)
            camera_movement: Camera movement preset
            lighting: Lighting preset
            color_grading: Color grading preset  
            duration: Video duration in seconds
            fps: Frames per second
            resolution: Video resolution (width, height)
            seed: Random seed for reproducibility
            output_path: Output video file path
            
        Returns:
            Path to generated video file
        """
        print(f"🎬 Generating cinematic video: {prompt[:50]}...")
        
        # Style parameters for prompt enhancement
        style_params = {
            "cinematic_style": style,
            "camera_movement": camera_movement,
            "lighting": lighting,
            "color_grading": color_grading
        }
        
        # Enhance prompt with cinematic elements
        enhanced_prompt = self._enhance_prompt_with_cinematic_elements(prompt, style_params)
        print(f"Enhanced prompt: {enhanced_prompt[:100]}...")
        
        # Calculate frames based on duration and fps
        num_frames = int(duration * fps)
        
        # Check memory and adjust settings if needed
        memory_info = self.monitor_memory()
        if memory_info.get("memory_percent", 0) > 80:
            print("High memory usage detected, reducing resolution...")
            resolution = (min(resolution[0], 384), min(resolution[1], 384))
            fps = max(fps, 6)  # Reduce fps if needed
        
        # Generation parameters
        gen_params = {
            "num_frames": num_frames,
            "fps": fps,
            "guidance_scale": 7.5,
            "inference_steps": 25,  # Higher quality for cinematic
            "resolution": resolution,
            "seed": seed,
            "prompt": enhanced_prompt,
            "negative_prompt": "blurry, low quality, distorted, artifact, noise"
        }
        
        try:
            # Memory-efficient generation
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # This would integrate with the actual VideoCrafter2 model
                # For now, simulating the generation process
                frames = self._simulate_generation(gen_params)
            
            # Apply cinematic post-processing
            frames = self._apply_cinematic_postprocessing(frames, style_params)
            
            # Save the video
            output_path = self.save_video(frames, output_path, fps)
            
            # Generate metadata
            self._save_metadata(output_path, prompt, style_params, gen_params)
            
            print(f"✅ Cinematic video generated: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Error generating cinematic video: {e}")
            raise
    
    def _simulate_generation(self, params: Dict) -> np.ndarray:
        """Simulate video generation process (replace with actual model inference)."""
        # This would call the actual VideoCrafter2 model
        print(f"Generating {params['num_frames']} frames at {params['resolution']}...")
        
        # Simulate processing time
        import time
        time.sleep(2)  # Simulate generation time
        
        # Create dummy frames for demonstration
        frames = np.random.rand(
            params['num_frames'], 
            params['resolution'][1], 
            params['resolution'][0], 
            3
        )
        
        return frames
    
    def _apply_cinematic_postprocessing(self, frames: np.ndarray, style_params: Dict) -> np.ndarray:
        """Apply cinematic post-processing effects."""
        print("Applying cinematic post-processing...")
        
        # Apply color grading
        color_grade = self.color_presets.get(style_params.get("color_grading", "cinematic"))
        if color_grade:
            # Apply contrast and saturation adjustments
            color_params = color_grade["color_params"]
            
            # Simple color grading (would be more sophisticated in real implementation)
            contrast = color_params.get("contrast", 1.0)
            saturation = color_params.get("saturation", 1.0)
            
            frames = frames * contrast
            frames = np.clip(frames, 0, 1)
            
            # Convert to HSV for saturation adjustment
            frames_hsv = self._rgb_to_hsv(frames)
            frames_hsv[:, :, :, 1] *= saturation  # Adjust saturation
            frames_hsv[:, :, :, 1] = np.clip(frames_hsv[:, :, :, 1], 0, 1)
            frames = self._hsv_to_rgb(frames_hsv)
        
        # Apply depth of field effect (simplified)
        if style_params.get("depth_of_field", True):
            frames = self._apply_depth_of_field(frames)
        
        return frames
    
    def _rgb_to_hsv(self, frames: np.ndarray) -> np.ndarray:
        """Convert RGB to HSV color space."""
        frames = np.clip(frames, 0, 1)
        return np.stack([cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2HSV) for frame in frames])
    
    def _hsv_to_rgb(self, frames: np.ndarray) -> np.ndarray:
        """Convert HSV to RGB color space.""" 
        return np.stack([cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_HSV2RGB) for frame in frames]) / 255.0
    
    def _apply_depth_of_field(self, frames: np.ndarray) -> np.ndarray:
        """Apply depth of field blur effect (simplified version)."""
        # This is a simplified DoF effect
        # Real implementation would use depth maps or segmentation
        return frames
    
    def _save_metadata(self, video_path: str, prompt: str, style_params: Dict, gen_params: Dict):
        """Save generation metadata."""
        metadata = {
            "prompt": prompt,
            "style_params": style_params,
            "generation_params": gen_params,
            "generator_version": "1.0.0",
            "timestamp": "2025-10-27T14:20:01Z"
        }
        
        metadata_path = video_path.replace('.mp4', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_available_presets(self) -> Dict[str, Any]:
        """Get all available presets for this generator."""
        return {
            "camera_movements": self.camera_presets,
            "lighting": self.lighting_presets, 
            "color_grading": self.color_presets,
            "styles": {
                "hollywood": "Professional Hollywood-style cinematography",
                "indie": "Independent film aesthetic",
                "arthouse": "Experimental/art house cinematography"
            }
        }
    
    def create_story_sequence(self, 
                             prompts: List[str], 
                             transitions: List[str] = None,
                             output_path: str = "outputs/cinematic_sequence.mp4") -> str:
        """
        Generate a sequence of cinematic clips for storytelling.
        
        Args:
            prompts: List of scene descriptions
            transitions: List of transition types between scenes
            output_path: Output video file path
            
        Returns:
            Path to the complete sequence video
        """
        print(f"🎬 Generating cinematic sequence with {len(prompts)} scenes...")
        
        clips = []
        for i, prompt in enumerate(prompts):
            print(f"Generating scene {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            clip_path = f"outputs/temp_scene_{i}.mp4"
            self.generate(
                prompt=prompt,
                camera_movement="static",  # Could vary by scene
                output_path=clip_path
            )
            clips.append(clip_path)
        
        # Concatenate clips (would use moviepy or ffmpeg)
        final_path = self._concatenate_videos(clips, output_path)
        
        # Clean up temporary files
        for clip in clips:
            if os.path.exists(clip):
                os.remove(clip)
        
        print(f"✅ Cinematic sequence completed: {final_path}")
        return final_path
    
    def _concatenate_videos(self, video_paths: List[str], output_path: str) -> str:
        """Concatenate multiple video clips into one."""
        # This would use ffmpeg or moviepy to concatenate videos
        # For now, just copy the first video
        if video_paths:
            import shutil
            shutil.copy2(video_paths[0], output_path)
        return output_path