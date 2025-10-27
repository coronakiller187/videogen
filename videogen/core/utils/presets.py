"""
Preset Management for VideoGen
Handles style presets, camera movements, and reusable configurations
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path


class PresetManager:
    """
    Manager for video generation presets and configurations.
    Handles loading, saving, and applying preset configurations.
    """
    
    def __init__(self, preset_dir: Optional[str] = None):
        """
        Initialize preset manager.
        
        Args:
            preset_dir: Directory to store presets (None for default)
        """
        if preset_dir:
            self.preset_dir = Path(preset_dir)
        else:
            self.preset_dir = Path.home() / ".videogen" / "presets"
        
        self.preset_dir.mkdir(parents=True, exist_ok=True)
        
        # Load built-in presets
        self._load_builtin_presets()
        
        # Load user presets
        self._load_user_presets()
        
        print(f"🎨 Preset Manager initialized with {self.get_total_preset_count()} presets")
    
    def _load_builtin_presets(self):
        """Load built-in presets for all generator types."""
        
        # Camera movement presets
        self.presets = {
            "camera_movements": {
                "static": {
                    "name": "Static Shot",
                    "description": "No camera movement, stable composition",
                    "params": {
                        "dolly": 0.0,
                        "pan": 0.0,
                        "tilt": 0.0,
                        "roll": 0.0,
                        "speed": 0.0
                    },
                    "tags": ["stable", "locked"]
                },
                "slow_dolly_in": {
                    "name": "Slow Dolly In",
                    "description": "Gradual push toward subject",
                    "params": {
                        "dolly": 1.0,
                        "pan": 0.0,
                        "tilt": 0.0,
                        "roll": 0.0,
                        "speed": 0.3,
                        "easing": "smooth_in"
                    },
                    "tags": ["push_in", "engaging", "cinematic"]
                },
                "dolly_out": {
                    "name": "Dolly Out",
                    "description": "Pull back from subject to reveal environment",
                    "params": {
                        "dolly": -1.0,
                        "pan": 0.0,
                        "tilt": 0.0,
                        "roll": 0.0,
                        "speed": 0.4,
                        "easing": "smooth_out"
                    },
                    "tags": ["reveal", "epic"]
                },
                "orbital": {
                    "name": "Orbital Movement",
                    "description": "Camera orbits around subject",
                    "params": {
                        "dolly": 0.0,
                        "pan": 1.0,
                        "tilt": 0.2,
                        "roll": 0.0,
                        "speed": 0.5,
                        "easing": "constant_speed",
                        "altitude": 0.3
                    },
                    "tags": ["dynamic", "showcase", "product"]
                },
                "tracking_left": {
                    "name": "Tracking Left",
                    "description": "Camera follows action from left",
                    "params": {
                        "dolly": 0.0,
                        "pan": -0.8,
                        "tilt": 0.0,
                        "roll": 0.0,
                        "speed": 0.4,
                        "easing": "smooth_follow"
                    },
                    "tags": ["follow", "dynamic"]
                },
                "crane_up": {
                    "name": "Crane Up",
                    "description": "Camera rises to reveal larger scene",
                    "params": {
                        "dolly": 0.0,
                        "pan": 0.0,
                        "tilt": 1.0,
                        "roll": 0.0,
                        "speed": 0.3,
                        "easing": "smooth_rise"
                    },
                    "tags": ["reveal", "epic"]
                }
            },
            
            # Lighting presets
            "lighting": {
                "natural": {
                    "name": "Natural Light",
                    "description": "Realistic daylight illumination",
                    "params": {
                        "primary_light": "soft_directional",
                        "fill_light": "ambient",
                        "rim_light": None,
                        "color_temperature": 5600,
                        "intensity": 1.0,
                        "shadows": "soft"
                    },
                    "tags": ["realistic", "outdoor"]
                },
                "golden_hour": {
                    "name": "Golden Hour",
                    "description": "Warm sunset/sunrise lighting",
                    "params": {
                        "primary_light": "directional_warm",
                        "fill_light": "warm_ambient",
                        "rim_light": "warm_back",
                        "color_temperature": 3200,
                        "intensity": 1.2,
                        "shadows": "warm"
                    },
                    "tags": ["warm", "romantic", "cinematic"]
                },
                "dramatic": {
                    "name": "Dramatic Lighting",
                    "description": "High contrast dramatic lighting",
                    "params": {
                        "primary_light": "hard_directional",
                        "fill_light": "minimal",
                        "rim_light": "strong_back",
                        "color_temperature": 5600,
                        "intensity": 1.4,
                        "shadows": "hard",
                        "contrast": 1.3
                    },
                    "tags": ["cinematic", "moody", "intense"]
                },
                "neon": {
                    "name": "Neon/Cyberpunk",
                    "description": "Vibrant neon lighting for sci-fi",
                    "params": {
                        "primary_light": "neon_pink",
                        "fill_light": "neon_cyan",
                        "rim_light": "neon_purple",
                        "color_temperature": 0,  # RGB mode
                        "neon_colors": ["#FF0080", "#00FFFF", "#8000FF"],
                        "intensity": 1.5,
                        "shadows": "colorful"
                    },
                    "tags": ["cyberpunk", "futuristic", "vibrant"]
                }
            },
            
            # Color grading presets
            "color_grading": {
                "cinematic": {
                    "name": "Cinematic",
                    "description": "Professional film color grade",
                    "params": {
                        "contrast": 1.2,
                        "saturation": 0.9,
                        "shadows": -0.1,
                        "highlights": 0.1,
                        "vignette": 0.15,
                        "grain": 0.05,
                        "lut": "filmic"
                    },
                    "tags": ["professional", "film"]
                },
                "noir": {
                    "name": "Film Noir",
                    "description": "Classic black and white treatment",
                    "params": {
                        "contrast": 1.5,
                        "saturation": 0.0,
                        "shadows": -0.2,
                        "highlights": 0.2,
                        "vignette": 0.3,
                        "grain": 0.1,
                        "sepia": 0.0
                    },
                    "tags": ["classic", "dramatic"]
                },
                "vibrant": {
                    "name": "Vibrant",
                    "description": "High saturation, punchy colors",
                    "params": {
                        "contrast": 1.1,
                        "saturation": 1.3,
                        "shadows": 0.0,
                        "highlights": 0.1,
                        "vignette": 0.0,
                        "grain": 0.0
                    },
                    "tags": ["colorful", "energetic"]
                }
            },
            
            # Product animation presets
            "product_animations": {
                "turntable_360": {
                    "name": "360° Turntable",
                    "description": "Full rotation showcase",
                    "params": {
                        "duration": 4.0,
                        "frames_per_rotation": 48,
                        "easing": "constant_speed",
                        "pause_frames": 8
                    },
                    "tags": ["360", "showcase"]
                },
                "hero_slide": {
                    "name": "Hero Slide In",
                    "description": "Product slides into frame",
                    "params": {
                        "duration": 3.0,
                        "entry_direction": "left",
                        "easing": "smooth_in"
                    },
                    "tags": ["dynamic", "attention"]
                }
            },
            
            # Product backgrounds
            "product_backgrounds": {
                "studio_white": {
                    "name": "Studio White",
                    "description": "Clean white background",
                    "params": {
                        "color": "#FFFFFF",
                        "lighting": "soft_shadow",
                        "infinite": True
                    },
                    "tags": ["clean", "professional"]
                },
                "infinite_white": {
                    "name": "Infinite White",
                    "description": "Seamless white background",
                    "params": {
                        "color": "#FFFFFF",
                        "lighting": "even_distributed",
                        "infinite": True
                    },
                    "tags": ["seamless", "studio"]
                }
            },
            
            # Animation styles for explainers
            "animation_styles": {
                "flat_2d": {
                    "name": "Flat 2D",
                    "description": "Simple flat animation",
                    "params": {
                        "shading": False,
                        "textures": False,
                        "gradients": True,
                        "line_weight": "thin"
                    },
                    "tags": ["clean", "simple"]
                },
                "cartoon": {
                    "name": "Cartoon",
                    "description": "Playful cartoon style",
                    "params": {
                        "shading": True,
                        "textures": False,
                        "gradients": True,
                        "line_weight": "bold"
                    },
                    "tags": ["playful", "friendly"]
                }
            },
            
            # Educational styles
            "explainer_styles": {
                "scientific": {
                    "name": "Scientific",
                    "description": "Academic/scientific presentation",
                    "params": {
                        "color_palette": ["#2E86AB", "#A23B72", "#F18F01"],
                        "font": "clean_sans",
                        "icons": "outline"
                    },
                    "tags": ["professional", "detailed"]
                },
                "corporate": {
                    "name": "Corporate",
                    "description": "Business presentation style",
                    "params": {
                        "color_palette": ["#1B4F72", "#0B5345", "#D35400"],
                        "font": "professional",
                        "icons": "filled"
                    },
                    "tags": ["business", "clean"]
                }
            }
        }
    
    def _load_user_presets(self):
        """Load user-defined presets from preset directory."""
        user_presets_file = self.preset_dir / "user_presets.yaml"
        
        if user_presets_file.exists():
            try:
                with open(user_presets_file, 'r') as f:
                    user_presets = yaml.safe_load(f)
                
                # Merge user presets with built-in ones
                for category, presets in user_presets.items():
                    if category not in self.presets:
                        self.presets[category] = {}
                    self.presets[category].update(presets)
                
                print(f"✅ Loaded user presets from {user_presets_file}")
                
            except Exception as e:
                print(f"⚠️ Failed to load user presets: {e}")
    
    def get_presets(self, category: str) -> Dict[str, Any]:
        """
        Get presets for a specific category.
        
        Args:
            category: Preset category name
            
        Returns:
            Dictionary of presets in the category
        """
        return self.presets.get(category, {})
    
    def get_preset(self, category: str, preset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific preset.
        
        Args:
            category: Preset category
            preset_name: Name of the preset
            
        Returns:
            Preset configuration or None if not found
        """
        category_presets = self.get_presets(category)
        return category_presets.get(preset_name)
    
    def list_presets(self, category: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List all available presets.
        
        Args:
            category: Optional category filter
            
        Returns:
            Dictionary mapping categories to preset names
        """
        if category:
            return {category: list(self.get_presets(category).keys())}
        else:
            return {cat: list(presets.keys()) for cat, presets in self.presets.items()}
    
    def create_preset(self, category: str, name: str, config: Dict[str, Any]) -> bool:
        """
        Create a new preset.
        
        Args:
            category: Preset category
            name: Preset name
            config: Preset configuration
            
        Returns:
            True if successful
        """
        try:
            # Load existing user presets
            user_presets_file = self.preset_dir / "user_presets.yaml"
            user_presets = {}
            
            if user_presets_file.exists():
                with open(user_presets_file, 'r') as f:
                    user_presets = yaml.safe_load(f) or {}
            
            # Add new preset
            if category not in user_presets:
                user_presets[category] = {}
            
            user_presets[category][name] = config
            
            # Save to file
            with open(user_presets_file, 'w') as f:
                yaml.dump(user_presets, f, default_flow_style=False)
            
            # Update in-memory presets
            if category not in self.presets:
                self.presets[category] = {}
            self.presets[category][name] = config
            
            print(f"✅ Created preset '{name}' in category '{category}'")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create preset: {e}")
            return False
    
    def apply_preset(self, category: str, preset_name: str, target_config: Dict[str, Any]) -> bool:
        """
        Apply a preset to a target configuration.
        
        Args:
            category: Preset category
            preset_name: Preset name
            target_config: Configuration to apply preset to
            
        Returns:
            True if successful
        """
        preset = self.get_preset(category, preset_name)
        
        if not preset:
            print(f"❌ Preset '{preset_name}' not found in category '{category}'")
            return False
        
        # Apply preset parameters
        preset_params = preset.get("params", {})
        target_config.update(preset_params)
        
        print(f"✅ Applied preset '{preset_name}' from category '{category}'")
        return True
    
    def combine_presets(self, combinations: List[tuple]) -> Dict[str, Any]:
        """
        Combine multiple presets into a single configuration.
        
        Args:
            combinations: List of (category, preset_name) tuples
            
        Returns:
            Combined configuration
        """
        combined_config = {}
        
        for category, preset_name in combinations:
            preset = self.get_preset(category, preset_name)
            if preset:
                # Merge parameters
                preset_params = preset.get("params", {})
                for key, value in preset_params.items():
                    combined_config[key] = value
            else:
                print(f"⚠️ Preset '{preset_name}' not found in category '{category}'")
        
        return combined_config
    
    def find_presets_by_tags(self, tags: List[str]) -> List[tuple]:
        """
        Find presets that match specific tags.
        
        Args:
            tags: List of tags to match
            
        Returns:
            List of (category, preset_name, preset) tuples
        """
        matches = []
        
        for category, category_presets in self.presets.items():
            for preset_name, preset in category_presets.items():
                preset_tags = preset.get("tags", [])
                if any(tag in preset_tags for tag in tags):
                    matches.append((category, preset_name, preset))
        
        return matches
    
    def export_presets(self, categories: List[str], export_path: str) -> bool:
        """
        Export specific preset categories to a file.
        
        Args:
            categories: List of categories to export
            export_path: Path to export file
            
        Returns:
            True if successful
        """
        try:
            export_data = {}
            
            for category in categories:
                if category in self.presets:
                    export_data[category] = self.presets[category]
                else:
                    print(f"⚠️ Category '{category}' not found")
            
            # Save to file
            with open(export_path, 'w') as f:
                if export_path.endswith('.json'):
                    json.dump(export_data, f, indent=2)
                else:
                    yaml.dump(export_data, f, default_flow_style=False)
            
            print(f"✅ Exported presets to {export_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to export presets: {e}")
            return False
    
    def get_recommended_presets(self, use_case: str) -> List[tuple]:
        """
        Get recommended presets for a specific use case.
        
        Args:
            use_case: Use case ("cinematic", "product", "explainer")
            
        Returns:
            List of recommended (category, preset_name) tuples
        """
        recommendations = {
            "cinematic": [
                ("camera_movements", "slow_dolly_in"),
                ("lighting", "dramatic"),
                ("color_grading", "cinematic")
            ],
            "product": [
                ("product_animations", "turntable_360"),
                ("product_backgrounds", "studio_white"),
                ("lighting", "natural")
            ],
            "explainer": [
                ("animation_styles", "flat_2d"),
                ("explainer_styles", "scientific")
            ]
        }
        
        return recommendations.get(use_case, [])
    
    def get_total_preset_count(self) -> int:
        """Get total number of available presets."""
        return sum(len(presets) for presets in self.presets.values())
    
    def search_presets(self, query: str) -> List[Dict[str, str]]:
        """
        Search presets by name, description, or tags.
        
        Args:
            query: Search query
            
        Returns:
            List of matching presets with metadata
        """
        query_lower = query.lower()
        matches = []
        
        for category, category_presets in self.presets.items():
            for preset_name, preset in category_presets.items():
                # Search in name, description, and tags
                searchable_text = (
                    preset_name.lower() + " " +
                    preset.get("description", "").lower() + " " +
                    " ".join(preset.get("tags", [])).lower()
                )
                
                if query_lower in searchable_text:
                    matches.append({
                        "category": category,
                        "name": preset_name,
                        "description": preset.get("description", ""),
                        "tags": preset.get("tags", [])
                    })
        
        return matches