#!/usr/bin/env python3
"""
VideoGen Demo Script
Demonstrates all generator types with example prompts
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core import (
    CinematicGenerator, 
    ProductGenerator, 
    ExplainerGenerator,
    PresetManager
)


def demo_cinematic_generation():
    """Demonstrate cinematic video generation."""
    print("🎬 DEMO: Cinematic Video Generation")
    print("-" * 40)
    
    with CinematicGenerator() as gen:
        # Show available presets
        presets = gen.get_available_presets()
        print(f"Available camera movements: {len(presets['camera_movements'])}")
        print(f"Available lighting styles: {len(presets['lighting'])}")
        print(f"Available color grading: {len(presets['color_grading'])}")
        
        # Example prompts for different styles
        examples = [
            {
                "prompt": "A lone astronaut walking on Mars at sunset, cinematic composition",
                "style": "hollywood",
                "camera_movement": "slow_dolly_in",
                "lighting": "golden_hour",
                "color_grading": "cinematic"
            },
            {
                "prompt": "Cyberpunk street scene with neon lights and flying cars",
                "style": "indie", 
                "camera_movement": "tracking_right",
                "lighting": "neon",
                "color_grading": "vibrant"
            }
        ]
        
        print(f"\nGenerating {len(examples)} cinematic examples...")
        for i, example in enumerate(examples):
            print(f"  {i+1}. {example['prompt'][:50]}...")
            # Simulate generation (in real implementation, would generate actual video)
            print(f"     ✅ Would generate: outputs/cinematic_demo_{i+1}.mp4")
    
    print("✅ Cinematic demo complete!\n")


def demo_product_generation():
    """Demonstrate product video generation."""
    print("📦 DEMO: Product Video Generation")
    print("-" * 40)
    
    with ProductGenerator() as gen:
        # Show available presets
        presets = gen.get_available_presets()
        print(f"Available animations: {len(presets['animations'])}")
        print(f"Available backgrounds: {len(presets['backgrounds'])}")
        print(f"Available styles: {len(presets['styles'])}")
        
        # Example product demonstrations
        examples = [
            {
                "description": "Luxury watch with turntable animation",
                "animation": "turntable_360",
                "style": "luxury",
                "background": "studio_black"
            },
            {
                "description": "Modern laptop with hero slide-in",
                "animation": "hero_slide",
                "style": "tech",
                "background": "studio_white"
            },
            {
                "description": "Coffee maker in lifestyle setting",
                "animation": "lifestyle_intro",
                "style": "lifestyle",
                "background": "lifestyle_kitchen"
            }
        ]
        
        print(f"\nGenerating {len(examples)} product examples...")
        for i, example in enumerate(examples):
            print(f"  {i+1}. {example['description']}")
            print(f"     ✅ Would generate: outputs/product_demo_{i+1}.mp4")
    
    print("✅ Product demo complete!\n")


def demo_explainer_generation():
    """Demonstrate explainer video generation."""
    print("📚 DEMO: Explainer Video Generation")
    print("-" * 40)
    
    with ExplainerGenerator() as gen:
        # Show available presets
        presets = gen.get_available_presets()
        print(f"Available animation styles: {len(presets['animation_styles'])}")
        print(f"Available educational presets: {len(presets['educational_presets'])}")
        print(f"Available visual elements: {len(presets['visual_elements'])}")
        
        # Example explainer topics
        examples = [
            {
                "topic": "How solar panels work",
                "style": "flat_2d",
                "preset": "scientific",
                "level": "general"
            },
            {
                "topic": "Introduction to machine learning",
                "style": "cartoon",
                "preset": "tech",
                "level": "professional"
            },
            {
                "topic": "Healthy eating for kids",
                "style": "cartoon",
                "preset": "kids",
                "level": "kids"
            }
        ]
        
        print(f"\nGenerating {len(examples)} explainer examples...")
        for i, example in enumerate(examples):
            print(f"  {i+1}. {example['topic']}")
            print(f"     ✅ Would generate: outputs/explainer_demo_{i+1}.mp4")
    
    print("✅ Explainer demo complete!\n")


def demo_preset_system():
    """Demonstrate preset management system."""
    print("🎨 DEMO: Preset System")
    print("-" * 40)
    
    preset_manager = PresetManager()
    
    # Show preset categories
    all_presets = preset_manager.list_presets()
    print(f"Available preset categories: {len(all_presets)}")
    for category, preset_names in all_presets.items():
        print(f"  {category}: {len(preset_names)} presets")
    
    # Search presets
    print("\nSearching for 'lighting' related presets...")
    lighting_presets = preset_manager.search_presets("lighting")
    for preset in lighting_presets[:3]:  # Show first 3
        print(f"  - {preset['category']}/{preset['name']}: {preset['description']}")
    
    # Find presets by tags
    print("\nFinding presets with 'cinematic' tag...")
    cinematic_presets = preset_manager.find_presets_by_tags(["cinematic"])
    for category, name, preset in cinematic_presets[:3]:  # Show first 3
        print(f"  - {name}: {preset['description']}")
    
    # Recommended presets
    print("\nRecommended presets for cinematic use case:")
    recommendations = preset_manager.get_recommended_presets("cinematic")
    for category, preset_name in recommendations:
        preset = preset_manager.get_preset(category, preset_name)
        print(f"  - {preset_name}: {preset['description']}")
    
    print("✅ Preset system demo complete!\n")


def demo_system_info():
    """Demonstrate system information and monitoring."""
    print("🖥️ DEMO: System Information")
    print("-" * 40)
    
    from core.utils.memory import MemoryManager
    from core.utils.config import Config
    import torch
    
    # Memory information
    if torch.cuda.is_available():
        memory_manager = MemoryManager(torch.device('cuda'))
        memory_info = memory_manager.get_memory_info()
        
        print("GPU Information:")
        for key, value in memory_info.items():
            if 'gpu' in key and isinstance(value, float):
                print(f"  {key}: {value:.1f}")
            elif 'gpu' in key and isinstance(value, str):
                print(f"  {key}: {value}")
    
    # Configuration
    config = Config()
    env_info = config.get_environment_info()
    
    print(f"\nEnvironment: {env_info['environment']}")
    if env_info['gpu_memory_gb'] > 0:
        print(f"GPU Memory: {env_info['gpu_memory_gb']:.1f}GB")
    
    optimal_settings = config.get_memory_optimized_settings()
    print(f"Optimal settings: {optimal_settings['quality_mode']} quality")
    
    print("✅ System info demo complete!\n")


def demo_batch_processing():
    """Demonstrate batch processing capabilities."""
    print("🔄 DEMO: Batch Processing")
    print("-" * 40)
    
    # Example batch processing scenarios
    batch_scenarios = [
        {
            "type": "cinematic",
            "prompts": [
                "Sunset over ocean waves",
                "Forest path in morning light", 
                "City skyline at night",
                "Mountain peak in clouds"
            ],
            "config": {
                "style": "hollywood",
                "camera_movement": "slow_dolly_in",
                "lighting": "natural",
                "color_grading": "cinematic",
                "duration": 3.0
            }
        },
        {
            "type": "explainer",
            "prompts": [
                "Explain photosynthesis",
                "How volcanoes work",
                "Solar system overview",
                "Water cycle explained"
            ],
            "config": {
                "animation_style": "flat_2d",
                "educational_preset": "scientific",
                "education_level": "general",
                "duration": 4.0
            }
        }
    ]
    
    for scenario in batch_scenarios:
        print(f"{scenario['type'].title()} batch ({len(scenario['prompts'])} videos):")
        for i, prompt in enumerate(scenario['prompts']):
            print(f"  {i+1}. {prompt[:40]}...")
        print(f"  ✅ Would process all {len(scenario['prompts'])} videos efficiently")
    
    print("✅ Batch processing demo complete!\n")


def main():
    """Main demo function."""
    print("🎬 VideoGen System Demo")
    print("=" * 60)
    print("Demonstrating AI video generation capabilities")
    print("=" * 60 + "\n")
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Run all demos
    demo_system_info()
    demo_preset_system()
    demo_cinematic_generation()
    demo_product_generation()
    demo_explainer_generation()
    demo_batch_processing()
    
    print("🎉 All demos completed!")
    print("\n📚 Next Steps:")
    print("1. Try the interactive web interface:")
    print("   python interfaces/gradio_ui.py")
    print("\n2. Use the command line interface:")
    print("   python interfaces/cli.py --help")
    print("\n3. Run the full Colab notebook:")
    print("   examples/notebooks/VideoGen_Complete_Setup.ipynb")
    print("\n4. Check the examples directory for more demos")


if __name__ == "__main__":
    main()