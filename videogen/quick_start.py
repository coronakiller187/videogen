#!/usr/bin/env python3
"""
VideoGen Quick Start Guide
Simple examples for getting started with video generation
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


def quick_cinematic_example():
    """Quick example of cinematic video generation."""
    print("🎬 Quick Cinematic Example")
    print("=" * 30)
    
    with CinematicGenerator() as gen:
        print("✅ Cinematic generator initialized")
        
        # Generate a simple cinematic video
        video_path = gen.generate(
            prompt="A peaceful sunrise over a mountain lake with gentle mist",
            style="hollywood",
            camera_movement="slow_dolly_in", 
            lighting="golden_hour",
            color_grading="cinematic",
            duration=4.0,
            output_path="outputs/quick_cinematic.mp4"
        )
        
        print(f"✅ Video would be saved to: {video_path}")


def quick_product_example():
    """Quick example of product video generation."""
    print("\n📦 Quick Product Example")
    print("=" * 30)
    
    with ProductGenerator() as gen:
        print("✅ Product generator initialized")
        
        # Generate a product demo
        video_path = gen.generate(
            prompt="A sleek smartphone rotating on a clean white surface",
            animation_type="turntable_360",
            commercial_style="tech",
            background="studio_white", 
            product_lighting="commercial",
            duration=3.0,
            output_path="outputs/quick_product.mp4"
        )
        
        print(f"✅ Video would be saved to: {video_path}")


def quick_explainer_example():
    """Quick example of explainer video generation."""
    print("\n📚 Quick Explainer Example")
    print("=" * 30)
    
    with ExplainerGenerator() as gen:
        print("✅ Explainer generator initialized")
        
        # Generate an explainer video
        video_path = gen.generate(
            prompt="Simple animation showing how plants use sunlight to make food",
            animation_style="flat_2d",
            educational_preset="scientific",
            character_consistency=True,
            education_level="kids",
            duration=5.0,
            output_path="outputs/quick_explainer.mp4"
        )
        
        print(f"✅ Video would be saved to: {video_path}")


def show_system_info():
    """Show system information and available options."""
    print("\n🖥️ System Information")
    print("=" * 30)
    
    from core.utils.memory import MemoryManager
    from core.utils.config import Config
    import torch
    
    # GPU info
    if torch.cuda.is_available():
        memory_manager = MemoryManager(torch.device('cuda'))
        memory_info = memory_manager.get_memory_info()
        gpu_name = memory_info.get('gpu_name', 'Unknown GPU')
        gpu_memory = memory_info.get('gpu_total_gb', 0)
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("⚠️ No GPU detected - CPU mode (slower)")
    
    # Preset info
    preset_manager = PresetManager()
    all_presets = preset_manager.list_presets()
    total_presets = sum(len(presets) for presets in all_presets.values())
    print(f"🎨 Presets available: {total_presets} across {len(all_presets)} categories")
    
    # Configuration
    config = Config()
    env_info = config.get_environment_info()
    print(f"☁️ Environment: {env_info['environment']}")
    
    print(f"📁 Output directory: {os.path.abspath('outputs')}")


def show_available_options():
    """Show available generator options."""
    print("\n⚙️ Available Generator Options")
    print("=" * 35)
    
    # Cinematic options
    with CinematicGenerator() as gen:
        presets = gen.get_available_presets()
        print("🎭 Cinematic Generator:")
        print(f"  Styles: {', '.join(presets['styles'].keys())}")
        print(f"  Camera movements: {', '.join(list(presets['camera_movements'].keys())[:3])}...")
        print(f"  Lighting: {', '.join(list(presets['lighting'].keys())[:3])}...")
    
    # Product options  
    with ProductGenerator() as gen:
        presets = gen.get_available_presets()
        print("\n📦 Product Generator:")
        print(f"  Animations: {', '.join(list(presets['animations'].keys())[:3])}...")
        print(f"  Styles: {', '.join(presets['styles'])}")
        print(f"  Backgrounds: {', '.join(list(presets['backgrounds'].keys())[:3])}...")
    
    # Explainer options
    with ExplainerGenerator() as gen:
        presets = gen.get_available_presets()
        print("\n📚 Explainer Generator:")
        print(f"  Animation styles: {', '.join(presets['animation_styles'].keys())}")
        print(f"  Educational presets: {', '.join(list(presets['educational_presets'].keys())[:3])}...")
        print(f"  Education levels: {', '.join(['kids', 'general', 'professional'])}")


def main():
    """Main quick start function."""
    print("🎬 VideoGen Quick Start Guide")
    print("=" * 50)
    print("Simple examples to get you started with AI video generation")
    print("=" * 50)
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    
    # Show system info
    show_system_info()
    
    # Show available options
    show_available_options()
    
    # Run quick examples
    print("\n🚀 Running Quick Examples")
    print("=" * 30)
    
    try:
        quick_cinematic_example()
        quick_product_example()
        quick_explainer_example()
        
        print("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during examples: {e}")
        print("This is normal during initial setup - the system will be fully functional once dependencies are installed")
    
    print("\n📚 Next Steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run full demo: python examples/demo.py")
    print("3. Launch web interface: python interfaces/gradio_ui.py")
    print("4. Use CLI: python interfaces/cli.py --help")
    print("5. Run Colab notebook: examples/notebooks/VideoGen_Complete_Setup.ipynb")
    
    print("\n💡 Tips:")
    print("- Start with lower resolution (384x384) for faster testing")
    print("- Use the preset system for consistent results")
    print("- Monitor GPU memory during generation")
    print("- Check the examples directory for more detailed usage")


if __name__ == "__main__":
    main()