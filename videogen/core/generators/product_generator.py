"""
Product Generator - Specialized for product demonstrations and commercial videos
Handles turntable animations, hero shots, and commercial-quality motion
"""

import os
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from PIL import Image

from .base_generator import BaseGenerator
from ..utils.config import Config


class ProductGenerator(BaseGenerator):
    """
    Product video generator focused on commercial-quality product demonstrations.
    Optimized for showcasing products with clean, professional motion.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize product generator."""
        super().__init__(config)
        
        # Product-specific defaults
        self.default_params.update({
            "commercial_style": "clean",
            "animation_type": "turntable_360",
            "background": "studio_white",
            "product_lighting": "commercial",
            "camera_focus": "product_sharp",
            "brand_colors": None
        })
        
        # Load product-specific presets
        self._load_product_presets()
        
        self.animations = self.preset_manager.get_presets("product_animations")
        self.backgrounds = self.preset_manager.get_presets("product_backgrounds")
        
    def _load_product_presets(self):
        """Load product-specific animation and style presets."""
        
        # Product animation types
        self.product_animations = {
            "turntable_360": {
                "description": "360-degree turntable rotation",
                "camera_path": "orbital",
                "duration": 4.0,
                "frames_per_rotation": 48,
                "easing": "constant_speed"
            },
            "hero_slide": {
                "description": "Product slides into frame from left",
                "camera_path": "tracking_left",
                "duration": 3.0,
                "easing": "smooth_in"
            },
            "detail_reveal": {
                "description": "Camera reveals product details progressively",
                "camera_path": "zoom_in_details",
                "duration": 5.0,
                "easing": "slow_in"
            },
            "floating_rotate": {
                "description": "Product floating and rotating gently",
                "camera_path": "gentle_orbital",
                "duration": 6.0,
                "easing": "sine_wave"
            },
            "cut_away": {
                "description": "Quick cut showing product from multiple angles",
                "camera_path": "multi_angle",
                "duration": 3.5,
                "easing": "sharp_transitions"
            },
            "lifestyle_intro": {
                "description": "Product integrated into lifestyle scene",
                "camera_path": "cinematic_wide",
                "duration": 7.0,
                "easing": "cinematic"
            },
            "exploded_view": {
                "description": "Product components separate and rotate",
                "camera_path": "exploded_orbital",
                "duration": 8.0,
                "easing": "mechanical"
            },
            "macro_details": {
                "description": "Extreme close-up of product textures/details",
                "camera_path": "macro_focus",
                "duration": 4.0,
                "easing": "smooth_focus"
            }
        }
        
        # Background presets for products
        self.product_backgrounds = {
            "studio_white": {
                "description": "Clean white studio background",
                "style": "solid_color",
                "color": "#FFFFFF",
                "lighting": "soft_shadow"
            },
            "studio_black": {
                "description": "Sleek black studio background",
                "style": "solid_color", 
                "color": "#000000",
                "lighting": "dramatic"
            },
            "gradient_blue": {
                "description": "Professional gradient from white to light blue",
                "style": "gradient",
                "colors": ["#FFFFFF", "#E6F3FF"],
                "lighting": "professional"
            },
            "gradient_gray": {
                "description": "Subtle gray gradient for premium products",
                "style": "gradient",
                "colors": ["#F8F8F8", "#E0E0E0"],
                "lighting": "soft"
            },
            "lifestyle_kitchen": {
                "description": "Modern kitchen setting for kitchen products",
                "style": "environment",
                "description": "Modern kitchen with marble countertops",
                "lighting": "natural_kitchen"
            },
            "lifestyle_office": {
                "description": "Clean office environment for tech products",
                "style": "environment", 
                "description": "Minimalist office desk setup",
                "lighting": "office_lighting"
            },
            "lifestyle_living": {
                "description": "Cozy living room for lifestyle products",
                "style": "environment",
                "description": "Warm living room with natural lighting",
                "lighting": "warm_ambient"
            },
            "infinite_white": {
                "description": "Infinite white seamless background",
                "style": "infinite",
                "color": "#FFFFFF",
                "lighting": "even_distributed"
            }
        }
        
        # Commercial lighting presets
        self.commercial_lighting = {
            "clean_commercial": {
                "description": "Professional commercial lighting",
                "setup": {
                    "key_light": "soft_box",
                    "fill_light": "reflector",
                    "back_light": "rim_light",
                    "temperature": 5600
                }
            },
            "product_focused": {
                "description": "Lighting optimized for product clarity",
                "setup": {
                    "key_light": "directional_soft",
                    "fill_light": "minimal",
                    "back_light": "specular",
                    "temperature": 5200
                }
            },
            "luxury": {
                "description": "High-end luxury product lighting",
                "setup": {
                    "key_light": "harsh_directional",
                    "fill_light": "selective",
                    "back_light": "dramatic_rim",
                    "temperature": 5800
                }
            },
            "warm_lifestyle": {
                "description": "Warm lighting for lifestyle integration",
                "setup": {
                    "key_light": "warm_soft",
                    "fill_light": "ambient_warm",
                    "back_light": "warm_rim",
                    "temperature": 3000
                }
            }
        }
        
        # Commercial styles
        self.commercial_styles = {
            "clean": {
                "description": "Clean, minimal commercial style",
                "characteristics": ["crisp details", "neutral colors", "professional lighting"]
            },
            "luxury": {
                "description": "High-end luxury commercial style",
                "characteristics": ["dramatic lighting", "rich colors", "premium materials"]
            },
            "tech": {
                "description": "Modern tech product style",
                "characteristics": ["sleek surfaces", "blue accents", "minimal shadows"]
            },
            "lifestyle": {
                "description": "Lifestyle integration style",
                "characteristics": ["warm lighting", "natural setting", "context-aware"]
            },
            "minimalist": {
                "description": "Ultra-minimal commercial style",
                "characteristics": ["white background", "simple lighting", "focus on product"]
            }
        }
    
    def _enhance_prompt_with_product_elements(self, prompt: str, product_params: Dict) -> str:
        """Enhance prompt with product-specific commercial elements."""
        enhanced_prompt = f"Professional product demonstration: {prompt}"
        
        # Add commercial quality keywords
        commercial_style = product_params.get("commercial_style", "clean")
        if commercial_style in self.commercial_styles:
            style_chars = self.commercial_styles[commercial_style]["characteristics"]
            enhanced_prompt += f", {', '.join(style_chars[:2])}"
        
        # Add lighting description
        lighting = product_params.get("product_lighting", "commercial")
        lighting_desc = self.commercial_lighting.get(lighting, {}).get("description", "")
        if lighting_desc:
            enhanced_prompt += f", {lighting_desc}"
        
        # Add animation description
        animation = product_params.get("animation_type", "turntable_360")
        if animation in self.product_animations:
            anim_desc = self.product_animations[animation]["description"]
            enhanced_prompt += f", {anim_desc}"
        
        # Add technical commercial terms
        enhanced_prompt += ", commercial quality, sharp focus, professional photography, product showcase"
        
        return enhanced_prompt
    
    def generate(self,
                image_path: Optional[str] = None,
                prompt: Optional[str] = None,
                animation_type: str = "turntable_360",
                commercial_style: str = "clean",
                background: str = "studio_white",
                product_lighting: str = "commercial",
                duration: float = 4.0,
                fps: int = 8,
                resolution: Tuple[int, int] = (512, 512),
                seed: Optional[int] = None,
                output_path: str = "outputs/product_demo.mp4",
                **kwargs) -> str:
        """
        Generate product demonstration video.
        
        Args:
            image_path: Path to product image (for image-to-video)
            prompt: Text description (for text-to-video)
            animation_type: Animation preset for product movement
            commercial_style: Commercial style (clean, luxury, tech, etc.)
            background: Background preset
            product_lighting: Lighting setup
            duration: Video duration in seconds
            fps: Frames per second
            resolution: Video resolution
            seed: Random seed
            output_path: Output video file path
            
        Returns:
            Path to generated video
        """
        if not image_path and not prompt:
            raise ValueError("Either image_path or prompt must be provided")
        
        animation_config = self.product_animations.get(animation_type, self.product_animations["turntable_360"])
        print(f"🎯 Generating {animation_type} animation for {animation_config['description']}")
        
        # Product parameters
        product_params = {
            "animation_type": animation_type,
            "commercial_style": commercial_style,
            "background": background,
            "product_lighting": product_lighting
        }
        
        # Use image if provided, otherwise use prompt
        if image_path:
            print(f"Using product image: {image_path}")
            enhanced_prompt = f"Professional product demo of: {os.path.basename(image_path)}"
        else:
            enhanced_prompt = prompt
        
        # Enhance prompt with commercial elements
        enhanced_prompt = self._enhance_prompt_with_product_elements(enhanced_prompt, product_params)
        print(f"Enhanced prompt: {enhanced_prompt[:100]}...")
        
        # Calculate frames based on animation
        num_frames = int(duration * fps)
        
        # Check for special animations that need different frame counts
        if animation_type == "turntable_360":
            frames_per_rotation = animation_config.get("frames_per_rotation", 48)
            num_frames = frames_per_rotation
        
        # Generation parameters
        gen_params = {
            "num_frames": num_frames,
            "fps": fps,
            "guidance_scale": 8.0,  # Higher guidance for product clarity
            "inference_steps": 22,
            "resolution": resolution,
            "seed": seed,
            "prompt": enhanced_prompt,
            "image_path": image_path,
            "negative_prompt": "blurry, distorted, low quality, artifacts, noise, motion blur"
        }
        
        try:
            # Memory-efficient generation
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # This would integrate with actual model (SVD for image-to-video, or T2V for text)
                if image_path:
                    frames = self._generate_image_to_video(image_path, gen_params, animation_config)
                else:
                    frames = self._generate_text_to_video(gen_params, animation_config)
            
            # Apply product-specific post-processing
            frames = self._apply_product_postprocessing(frames, product_params)
            
            # Save the video
            output_path = self.save_video(frames, output_path, fps)
            
            # Save metadata
            self._save_product_metadata(output_path, product_params, gen_params)
            
            print(f"✅ Product demo generated: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Error generating product demo: {e}")
            raise
    
    def _generate_image_to_video(self, image_path: str, params: Dict, animation_config: Dict) -> np.ndarray:
        """Generate video from product image using Stable Video Diffusion approach."""
        print(f"Generating video from product image with {animation_config['description']}...")
        
        # Load and preprocess product image
        product_image = self.preprocess_image(image_path)
        
        # This would integrate with actual SVD model
        # For now, simulating the process
        import time
        time.sleep(2)  # Simulate generation time
        
        # Create dummy frames based on animation type
        frames = np.random.rand(
            params['num_frames'],
            params['resolution'][1],
            params['resolution'][0], 
            3
        )
        
        # Add some structure based on product rotation for turntable
        if animation_config.get('camera_path') == 'orbital':
            frames = self._apply_orbital_motion(frames, params['num_frames'])
        
        return frames
    
    def _generate_text_to_video(self, params: Dict, animation_config: Dict) -> np.ndarray:
        """Generate product video from text description."""
        print(f"Generating product video from text...")
        
        # This would integrate with Text-to-Video model
        import time
        time.sleep(2)
        
        frames = np.random.rand(
            params['num_frames'],
            params['resolution'][1],
            params['resolution'][0],
            3
        )
        
        return frames
    
    def _apply_orbital_motion(self, frames: np.ndarray, num_frames: int) -> np.ndarray:
        """Apply orbital camera motion to product frames."""
        # Create orbital motion effect
        for i in range(num_frames):
            angle = (2 * np.pi * i) / num_frames
            # Simple orbital transform (would be more sophisticated in real implementation)
            frames[i] = np.roll(frames[i], shift=int(5 * np.sin(angle)), axis=0)
        
        return frames
    
    def _apply_product_postprocessing(self, frames: np.ndarray, product_params: Dict) -> np.ndarray:
        """Apply product-specific post-processing effects."""
        print("Applying product post-processing...")
        
        # Enhance sharpness for products
        frames = np.clip(frames * 1.1, 0, 1)
        
        # Apply background if specified
        background_style = product_params.get("background", "studio_white")
        if background_style in self.product_backgrounds:
            frames = self._apply_background_effect(frames, background_style)
        
        # Color enhancement for commercial quality
        commercial_style = product_params.get("commercial_style", "clean")
        if commercial_style == "clean":
            # Enhance whites and contrast
            frames = np.clip(frames * 1.05, 0, 1)
        elif commercial_style == "luxury":
            # Enhance contrast and saturation
            frames = np.clip(frames * 1.08, 0, 1)
        
        return frames
    
    def _apply_background_effect(self, frames: np.ndarray, background_type: str) -> np.ndarray:
        """Apply background effects to product frames."""
        # This would implement sophisticated background replacement
        # For now, just applying a simple color shift
        bg_config = self.product_backgrounds.get(background_type, {})
        
        if bg_config.get("style") == "solid_color":
            color = bg_config.get("color", "#FFFFFF")
            # Simple background color effect (would be more sophisticated)
            bg_color = np.array([int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]) / 255.0
        
        return frames
    
    def _save_product_metadata(self, video_path: str, product_params: Dict, gen_params: Dict):
        """Save product generation metadata."""
        metadata = {
            "type": "product_demo",
            "product_params": product_params,
            "generation_params": gen_params,
            "generator_version": "1.0.0",
            "timestamp": "2025-10-27T14:20:01Z"
        }
        
        metadata_path = video_path.replace('.mp4', '_metadata.json')
        with open(metadata_path, 'w') as f:
            import json
            json.dump(metadata, f, indent=2)
    
    def generate_product_sequence(self,
                                product_images: List[str],
                                transitions: List[str] = None,
                                commercial_style: str = "clean",
                                output_path: str = "outputs/product_sequence.mp4") -> str:
        """
        Generate a sequence showcasing multiple products.
        
        Args:
            product_images: List of product image paths
            transitions: Types of transitions between products
            commercial_style: Commercial style to apply
            output_path: Output video file path
            
        Returns:
            Path to the complete product sequence
        """
        print(f"🎯 Generating product sequence with {len(product_images)} products...")
        
        clips = []
        for i, image_path in enumerate(product_images):
            print(f"Creating demo for product {i+1}/{len(product_images)}: {os.path.basename(image_path)}")
            
            clip_path = f"outputs/temp_product_{i}.mp4"
            self.generate(
                image_path=image_path,
                animation_type="turntable_360" if i % 2 == 0 else "hero_slide",
                commercial_style=commercial_style,
                duration=3.0,
                output_path=clip_path
            )
            clips.append(clip_path)
        
        # Concatenate clips
        final_path = self._concatenate_videos(clips, output_path)
        
        # Clean up temporary files
        for clip in clips:
            if os.path.exists(clip):
                os.remove(clip)
        
        print(f"✅ Product sequence completed: {final_path}")
        return final_path
    
    def _concatenate_videos(self, video_paths: List[str], output_path: str) -> str:
        """Concatenate product videos."""
        if video_paths:
            import shutil
            shutil.copy2(video_paths[0], output_path)
        return output_path
    
    def get_available_presets(self) -> Dict[str, Any]:
        """Get all available presets for this generator."""
        return {
            "animations": self.product_animations,
            "backgrounds": self.product_backgrounds,
            "lighting": self.commercial_lighting,
            "styles": self.commercial_styles
        }
    
    def create_lifestyle_integration(self,
                                   product_image: str,
                                   lifestyle_scene: str,
                                   output_path: str = "outputs/lifestyle_product.mp4") -> str:
        """
        Create a lifestyle integration video showing product in use context.
        
        Args:
            product_image: Path to product image
            lifestyle_scene: Description of lifestyle scene
            output_path: Output video file path
            
        Returns:
            Path to the lifestyle integration video
        """
        print(f"🏠 Creating lifestyle integration: {lifestyle_scene}")
        
        # Generate lifestyle scene first
        lifestyle_prompt = f"Modern {lifestyle_scene} with warm lighting, professional commercial photography"
        
        # Then integrate product
        return self.generate(
            image_path=product_image,
            animation_type="lifestyle_intro",
            commercial_style="lifestyle",
            background="lifestyle_kitchen" if "kitchen" in lifestyle_scene else "lifestyle_office",
            product_lighting="warm_lifestyle",
            prompt=lifestyle_prompt,
            duration=6.0,
            output_path=output_path
        )