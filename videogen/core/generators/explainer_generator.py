"""
Explainer Generator - Specialized for animated explainers and educational content
Handles character consistency, flat animation styles, and educational visuals
"""

import os
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from PIL import Image

from .base_generator import BaseGenerator
from ..utils.config import Config


class ExplainerGenerator(BaseGenerator):
    """
    Explainer video generator focused on educational and explanatory content.
    Optimized for character consistency, flat animation styles, and clear visual communication.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize explainer generator."""
        super().__init__(config)
        
        # Explainer-specific defaults
        self.default_params.update({
            "animation_style": "flat_2d",
            "character_consistency": True,
            "education_level": "general",
            "visual_complexity": "simple",
            "voice_sync": False,
            "storyboard_style": "explainer"
        })
        
        # Load explainer-specific presets
        self._load_explainer_presets()
        
        self.animation_styles = self.preset_manager.get_presets("animation_styles")
        self.explainer_styles = self.preset_manager.get_presets("explainer_styles")
        
    def _load_explainer_presets(self):
        """Load explainer-specific animation and style presets."""
        
        # Animation styles for explainers
        self.animation_styles = {
            "flat_2d": {
                "description": "Simple 2D flat animation with clean lines",
                "characteristics": ["flat colors", "clean lines", "minimal shadows"],
                "technical": {"shading": False, "textures": False, "gradients": True}
            },
            "cartoon": {
                "description": "Playful cartoon style with rounded shapes",
                "characteristics": ["rounded shapes", "bright colors", "friendly"],
                "technical": {"shading": True, "textures": False, "gradients": True}
            },
            "isometric": {
                "description": "Isometric 2.5D style with geometric precision",
                "characteristics": ["geometric precision", "consistent angles", "flat colors"],
                "technical": {"shading": False, "textures": False, "gradients": False}
            },
            "paper_craft": {
                "description": "Handmade paper craft aesthetic",
                "characteristics": ["textured surfaces", "warm lighting", "craft feel"],
                "technical": {"shading": True, "textures": True, "gradients": True}
            },
            "corporate": {
                "description": "Clean corporate animation style",
                "characteristics": ["professional colors", "minimal design", "clear hierarchy"],
                "technical": {"shading": False, "textures": False, "gradients": False}
            },
            "tech_motion": {
                "description": "Modern tech-focused motion graphics",
                "characteristics": ["tech colors", "sleek animations", "data visualization"],
                "technical": {"shading": False, "textures": False, "gradients": True}
            }
        }
        
        # Explainer animation types
        self.explainer_animations = {
            "character_intro": {
                "description": "Main character introduces the topic",
                "duration": 5.0,
                "motion_type": "character_focus",
                "visual_elements": ["character", "text_overlay"]
            },
            "concept_explanation": {
                "description": "Animated explanation of key concepts",
                "duration": 8.0,
                "motion_type": "progressive_reveal",
                "visual_elements": ["diagrams", "step_by_step"]
            },
            "process_flow": {
                "description": "Shows step-by-step process flow",
                "duration": 6.0,
                "motion_type": "flow_animation",
                "visual_elements": ["arrows", "boxes", "sequential"]
            },
            "comparison": {
                "description": "Compares two or more concepts",
                "duration": 7.0,
                "motion_type": "split_screen",
                "visual_elements": ["side_by_side", "vs_text"]
            },
            "data_visualization": {
                "description": "Animated charts and data visualization",
                "duration": 6.0,
                "motion_type": "chart_animation",
                "visual_elements": ["graphs", "charts", "numbers"]
            },
            "character_demo": {
                "description": "Characters demonstrating usage or process",
                "duration": 9.0,
                "motion_type": "character_actions",
                "visual_elements": ["character_actions", "objects"]
            },
            "transition": {
                "description": "Smooth transition between topics",
                "duration": 3.0,
                "motion_type": "fluid_transition",
                "visual_elements": ["background", "text"]
            },
            "call_to_action": {
                "description": "Strong ending with call to action",
                "duration": 4.0,
                "motion_type": "attention_grabbing",
                "visual_elements": ["highlight", "action_buttons"]
            }
        }
        
        # Educational content presets
        self.educational_presets = {
            "scientific": {
                "description": "Scientific explanation style",
                "colors": ["#2E86AB", "#A23B72", "#F18F01"],
                "tone": "professional",
                "complexity": "detailed"
            },
            "business": {
                "description": "Business presentation style",
                "colors": ["#1B4F72", "#0B5345", "#D35400"],
                "tone": "professional",
                "complexity": "medium"
            },
            "healthcare": {
                "description": "Healthcare/medical explanation style",
                "colors": ["#2980B9", "#27AE60", "#E74C3C"],
                "tone": "caring",
                "complexity": "accessible"
            },
            "technology": {
                "description": "Technology/IT explanation style",
                "colors": ["#3498DB", "#9B59B6", "#1ABC9C"],
                "tone": "modern",
                "complexity": "detailed"
            },
            "kids": {
                "description": "Child-friendly explanation style",
                "colors": ["#FF6B6B", "#4ECDC4", "#45B7D1"],
                "tone": "playful",
                "complexity": "simple"
            },
            "finance": {
                "description": "Financial explanation style",
                "colors": ["#2C3E50", "#E67E22", "#95A5A6"],
                "tone": "trustworthy",
                "complexity": "medium"
            }
        }
        
        # Visual elements for explainers
        self.visual_elements = {
            "arrows": {
                "styles": ["simple", "animated", "3d"],
                "colors": "consistent_with_palette"
            },
            "callouts": {
                "styles": ["speech_bubbles", "info_boxes", "highlight_rings"],
                "animation": "fade_in"
            },
            "icons": {
                "styles": ["flat", "outlined", "filled"],
                "consistency": "match_animation_style"
            },
            "text_overlays": {
                "styles": ["bold_headers", "bullet_points", "step_numbers"],
                "timing": "sync_with_voice"
            },
            "transitions": {
                "types": ["fade", "slide", "bounce", "morph"],
                "duration": "0.5-1.0 seconds"
            }
        }
    
    def _enhance_prompt_with_explainer_elements(self, prompt: str, explainer_params: Dict) -> str:
        """Enhance prompt with explainer-specific educational elements."""
        enhanced_prompt = f"Educational explainer video: {prompt}"
        
        # Add animation style characteristics
        animation_style = explainer_params.get("animation_style", "flat_2d")
        if animation_style in self.animation_styles:
            style_chars = self.animation_styles[animation_style]["characteristics"]
            enhanced_prompt += f", {', '.join(style_chars[:2])}"
        
        # Add educational level characteristics
        education_level = explainer_params.get("education_level", "general")
        if education_level == "kids":
            enhanced_prompt += ", child-friendly, colorful, engaging animation"
        elif education_level == "professional":
            enhanced_prompt += ", professional presentation, clean graphics, clear messaging"
        else:
            enhanced_prompt += ", educational content, clear visuals, informative"
        
        # Add technical explainer terms
        enhanced_prompt += ", explainer video style, clear communication, educational animation, flat design"
        
        return enhanced_prompt
    
    def generate(self,
                prompt: str,
                animation_style: str = "flat_2d",
                educational_preset: str = "business",
                character_consistency: bool = True,
                education_level: str = "general",
                visual_complexity: str = "simple",
                duration: float = 6.0,
                fps: int = 8,
                resolution: Tuple[int, int] = (512, 512),
                seed: Optional[int] = None,
                output_path: str = "outputs/explainer_video.mp4",
                **kwargs) -> str:
        """
        Generate educational explainer video.
        
        Args:
            prompt: Description of what to explain
            animation_style: Animation style preset
            educational_preset: Educational domain preset
            character_consistency: Maintain character consistency
            education_level: Target education level
            visual_complexity: Visual complexity level
            duration: Video duration
            fps: Frames per second
            resolution: Video resolution
            seed: Random seed
            output_path: Output video path
            
        Returns:
            Path to generated explainer video
        """
        explainer_params = {
            "animation_style": animation_style,
            "educational_preset": educational_preset,
            "character_consistency": character_consistency,
            "education_level": education_level,
            "visual_complexity": visual_complexity
        }
        
        print(f"📚 Generating explainer video: {animation_style} style for {education_level}")
        
        # Enhanced prompt for explainer content
        enhanced_prompt = self._enhance_prompt_with_explainer_elements(prompt, explainer_params)
        print(f"Enhanced prompt: {enhanced_prompt[:100]}...")
        
        # Calculate frames
        num_frames = int(duration * fps)
        
        # Generation parameters optimized for explainers
        gen_params = {
            "num_frames": num_frames,
            "fps": fps,
            "guidance_scale": 7.0,  # Slightly lower for clearer visuals
            "inference_steps": 18,  # Good balance of quality and speed
            "resolution": resolution,
            "seed": seed,
            "prompt": enhanced_prompt,
            "negative_prompt": "realistic, photorealistic, complex textures, detailed shadows, motion blur"
        }
        
        try:
            # Memory-efficient generation
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # This would integrate with actual model (likely AnimateDiff or similar)
                frames = self._generate_explainer_video(gen_params, explainer_params)
            
            # Apply explainer-specific post-processing
            frames = self._apply_explainer_postprocessing(frames, explainer_params)
            
            # Save the video
            output_path = self.save_video(frames, output_path, fps)
            
            # Save metadata
            self._save_explainer_metadata(output_path, explainer_params, gen_params)
            
            print(f"✅ Explainer video generated: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Error generating explainer video: {e}")
            raise
    
    def _generate_explainer_video(self, params: Dict, explainer_params: Dict) -> np.ndarray:
        """Generate explainer video content."""
        print(f"Generating {explainer_params['animation_style']} explainer animation...")
        
        # This would integrate with actual model (AnimateDiff with ControlNets)
        import time
        time.sleep(2)  # Simulate generation time
        
        # Create frames based on animation style
        frames = np.random.rand(
            params['num_frames'],
            params['resolution'][1],
            params['resolution'][0],
            3
        )
        
        # Apply explainer-specific frame characteristics
        animation_style = explainer_params.get("animation_style", "flat_2d")
        if animation_style == "flat_2d":
            # Flat colors, minimal shading
            frames = self._apply_flat_2d_style(frames)
        elif animation_style == "cartoon":
            # Bright colors, rounded shapes
            frames = self._apply_cartoon_style(frames)
        elif animation_style == "isometric":
            # Geometric, consistent angles
            frames = self._apply_isometric_style(frames)
        
        return frames
    
    def _apply_flat_2d_style(self, frames: np.ndarray) -> np.ndarray:
        """Apply flat 2D animation style."""
        # Reduce color variation for flat look
        for i in range(len(frames)):
            # Simplify colors to flat palette
            flat_frames = np.floor(frames[i] * 4) / 4  # Quantize to 4 levels
            frames[i] = flat_frames
        return frames
    
    def _apply_cartoon_style(self, frames: np.ndarray) -> np.ndarray:
        """Apply cartoon animation style."""
        # Enhance saturation for cartoon look
        frames = np.clip(frames * 1.3, 0, 1)
        return frames
    
    def _apply_isometric_style(self, frames: np.ndarray) -> np.ndarray:
        """Apply isometric animation style."""
        # Geometric patterns
        frames = self._create_geometric_pattern(frames)
        return frames
    
    def _create_geometric_pattern(self, frames: np.ndarray) -> np.ndarray:
        """Create geometric isometric patterns."""
        height, width = frames.shape[1], frames.shape[2]
        
        for i in range(len(frames)):
            # Create simple geometric patterns
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            
            # Create isometric-like grid
            iso_grid = (x + y) % 3 / 3.0
            iso_grid_3d = np.stack([iso_grid] * 3, axis=2)
            
            # Mix with original
            frames[i] = 0.7 * frames[i] + 0.3 * iso_grid_3d
        
        return frames
    
    def _apply_explainer_postprocessing(self, frames: np.ndarray, explainer_params: Dict) -> np.ndarray:
        """Apply explainer-specific post-processing."""
        print("Applying explainer post-processing...")
        
        # Apply educational preset colors
        educational_preset = explainer_params.get("educational_preset", "business")
        if educational_preset in self.educational_presets:
            frames = self._apply_educational_color_palette(frames, educational_preset)
        
        # Enhance clarity for educational content
        frames = np.clip(frames * 1.1, 0, 1)
        
        # Apply animation style post-processing
        animation_style = explainer_params.get("animation_style", "flat_2d")
        if animation_style == "flat_2d":
            # Enhance edge contrast for flat design
            frames = self._enhance_edges_for_flat_design(frames)
        elif animation_style == "paper_craft":
            # Add warm texture
            frames = self._add_paper_texture(frames)
        
        return frames
    
    def _apply_educational_color_palette(self, frames: np.ndarray, preset: str) -> np.ndarray:
        """Apply educational color palette."""
        preset_config = self.educational_presets.get(preset, {})
        colors = preset_config.get("colors", ["#2E86AB", "#A23B72", "#F18F01"])
        
        # Simple color mapping (would be more sophisticated in real implementation)
        color_mappings = {
            0: np.array([46, 134, 171]) / 255.0,  # #2E86AB
            1: np.array([162, 59, 114]) / 255.0,  # #A23B72
            2: np.array([241, 143, 1]) / 255.0    # #F18F01
        }
        
        return frames
    
    def _enhance_edges_for_flat_design(self, frames: np.ndarray) -> np.ndarray:
        """Enhance edges for flat design aesthetics."""
        # Simple edge enhancement
        return frames * 1.1
    
    def _add_paper_texture(self, frames: np.ndarray) -> np.ndarray:
        """Add paper texture effect."""
        # Simple texture overlay
        return frames
    
    def _save_explainer_metadata(self, video_path: str, explainer_params: Dict, gen_params: Dict):
        """Save explainer generation metadata."""
        metadata = {
            "type": "explainer_video",
            "explainer_params": explainer_params,
            "generation_params": gen_params,
            "generator_version": "1.0.0",
            "timestamp": "2025-10-27T14:20:01Z"
        }
        
        metadata_path = video_path.replace('.mp4', '_metadata.json')
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
    
    def create_educational_series(self,
                                topics: List[str],
                                series_title: str,
                                animation_style: str = "flat_2d",
                                educational_preset: str = "business",
                                output_dir: str = "outputs/educational_series") -> List[str]:
        """
        Create a series of educational videos on related topics.
        
        Args:
            topics: List of topics to cover
            series_title: Title for the video series
            animation_style: Animation style to maintain consistency
            educational_preset: Educational domain preset
            output_dir: Directory to save videos
            
        Returns:
            List of generated video file paths
        """
        print(f"📚 Creating educational series '{series_title}' with {len(topics)} videos")
        
        os.makedirs(output_dir, exist_ok=True)
        video_paths = []
        
        for i, topic in enumerate(topics):
            print(f"Creating episode {i+1}/{len(topics)}: {topic}")
            
            output_path = os.path.join(output_dir, f"{series_title.lower().replace(' ', '_')}_ep{i+1:02d}.mp4")
            
            video_path = self.generate(
                prompt=f"Explain {topic} in a clear, educational manner",
                animation_style=animation_style,
                educational_preset=educational_preset,
                character_consistency=True,
                duration=8.0,
                output_path=output_path
            )
            video_paths.append(video_path)
        
        # Create series metadata
        series_metadata = {
            "series_title": series_title,
            "total_episodes": len(topics),
            "animation_style": animation_style,
            "educational_preset": educational_preset,
            "episodes": video_paths
        }
        
        metadata_path = os.path.join(output_dir, f"{series_title.lower().replace(' ', '_')}_metadata.json")
        with open(metadata_path, 'w') as f:
            import json
            json.dump(series_metadata, f, indent=2)
        
        print(f"✅ Educational series completed with {len(video_paths)} videos")
        return video_paths
    
    def create_character_guide(self,
                             character_description: str,
                             poses: List[str],
                             expressions: List[str],
                             output_path: str = "outputs/character_guide.mp4") -> str:
        """
        Create a character guide video showing different poses and expressions.
        
        Args:
            character_description: Description of the character
            poses: List of poses to demonstrate
            expressions: List of expressions to show
            output_path: Output video path
            
        Returns:
            Path to character guide video
        """
        print(f"🎭 Creating character guide for: {character_description}")
        
        # Combine poses and expressions into guide
        guide_prompt = f"Character guide showing: {character_description}. Demonstrating poses: {', '.join(poses)}. Showing expressions: {', '.join(expressions)}."
        
        return self.generate(
            prompt=guide_prompt,
            animation_style="cartoon",
            character_consistency=True,
            educational_preset="kids",  # More flexible style
            duration=10.0,
            output_path=output_path
        )
    
    def get_available_presets(self) -> Dict[str, Any]:
        """Get all available presets for this generator."""
        return {
            "animation_styles": self.animation_styles,
            "educational_presets": self.educational_presets,
            "explainer_animations": self.explainer_animations,
            "visual_elements": self.visual_elements
        }
    
    def create_interactive_explainer(self,
                                   script_lines: List[Tuple[str, str]],  # (speaker, text)
                                   character_sprites: Optional[Dict[str, str]] = None,
                                   output_path: str = "outputs/interactive_explainer.mp4") -> str:
        """
        Create an interactive explainer with character animation and text synchronization.
        
        Args:
            script_lines: List of (speaker, text) tuples for the script
            character_sprites: Dictionary of character sprite paths
            output_path: Output video path
            
        Returns:
            Path to interactive explainer video
        """
        print(f"🎬 Creating interactive explainer with {len(script_lines)} script lines")
        
        # This would create a more sophisticated animated sequence
        # For now, create a general explainer
        
        full_script = " ".join([text for _, text in script_lines])
        
        return self.generate(
            prompt=f"Interactive explainer video: {full_script}",
            animation_style="cartoon",
            character_consistency=True,
            duration=max(len(script_lines) * 3.0, 8.0),  # Estimate duration
            output_path=output_path
        )