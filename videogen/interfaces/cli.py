"""
Command Line Interface for VideoGen
Provides command-line access to all video generation capabilities
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent))

from core import (
    CinematicGenerator, 
    ProductGenerator, 
    ExplainerGenerator,
    PresetManager
)


class VideoGenCLI:
    """Command line interface for VideoGen."""
    
    def __init__(self):
        """Initialize CLI."""
        self.preset_manager = PresetManager()
        self.generators = {}
    
    def get_generator(self, generator_type: str):
        """Get or create generator instance."""
        if generator_type not in self.generators:
            if generator_type == "cinematic":
                self.generators[generator_type] = CinematicGenerator()
            elif generator_type == "product":
                self.generators[generator_type] = ProductGenerator()
            elif generator_type == "explainer":
                self.generators[generator_type] = ExplainerGenerator()
            else:
                raise ValueError(f"Unknown generator type: {generator_type}")
        
        return self.generators[generator_type]
    
    def list_presets(self, generator_type: Optional[str] = None):
        """List available presets."""
        if generator_type:
            presets = self.preset_manager.get_presets(f"{generator_type}_presets")
            print(f"\n{generator_type.title()} Presets:")
            for name, preset in presets.items():
                print(f"  {name}: {preset.get('description', 'No description')}")
        else:
            all_presets = self.preset_manager.list_presets()
            for category, preset_names in all_presets.items():
                print(f"\n{category.title()}:")
                for name in preset_names:
                    preset = self.preset_manager.get_preset(category, name)
                    if preset:
                        print(f"  {name}: {preset.get('description', 'No description')}")
    
    def generate_cinematic(self, args):
        """Generate cinematic video from CLI args."""
        print("🎬 Generating cinematic video...")
        
        gen = self.get_generator("cinematic")
        
        # Parse resolution
        if args.resolution:
            width, height = map(int, args.resolution.split('x'))
        else:
            width, height = 512, 512
        
        # Generate video
        output_path = args.output or f"outputs/cinematic_{hash(args.prompt) % 10000}.mp4"
        
        video_path = gen.generate(
            prompt=args.prompt,
            style=args.style,
            camera_movement=args.camera_movement,
            lighting=args.lighting,
            color_grading=args.color_grading,
            duration=args.duration,
            resolution=(width, height),
            seed=args.seed,
            output_path=output_path
        )
        
        print(f"✅ Cinematic video generated: {video_path}")
        return video_path
    
    def generate_product(self, args):
        """Generate product demo from CLI args."""
        print("📦 Generating product demo...")
        
        gen = self.get_generator("product")
        
        # Parse resolution
        if args.resolution:
            width, height = map(int, args.resolution.split('x'))
        else:
            width, height = 512, 512
        
        # Generate video
        output_path = args.output or f"outputs/product_{hash(str(args.image) + args.prompt) % 10000}.mp4"
        
        video_path = gen.generate(
            image_path=args.image,
            prompt=args.prompt,
            animation_type=args.animation,
            commercial_style=args.style,
            background=args.background,
            product_lighting=args.lighting,
            duration=args.duration,
            resolution=(width, height),
            seed=args.seed,
            output_path=output_path
        )
        
        print(f"✅ Product demo generated: {video_path}")
        return video_path
    
    def generate_explainer(self, args):
        """Generate explainer video from CLI args."""
        print("📚 Generating explainer video...")
        
        gen = self.get_generator("explainer")
        
        # Parse resolution
        if args.resolution:
            width, height = map(int, args.resolution.split('x'))
        else:
            width, height = 512, 512
        
        # Generate video
        output_path = args.output or f"outputs/explainer_{hash(args.prompt) % 10000}.mp4"
        
        video_path = gen.generate(
            prompt=args.prompt,
            animation_style=args.animation_style,
            educational_preset=args.educational_preset,
            character_consistency=args.character_consistency,
            education_level=args.education_level,
            duration=args.duration,
            resolution=(width, height),
            seed=args.seed,
            output_path=output_path
        )
        
        print(f"✅ Explainer video generated: {video_path}")
        return video_path
    
    def batch_generate(self, args):
        """Generate multiple videos in batch."""
        print(f"🔄 Generating {len(args.prompts)} videos in batch...")
        
        # Determine generator type
        generator_map = {
            "cinematic": self.generate_cinematic,
            "product": self.generate_product,
            "explainer": self.generate_explainer
        }
        
        if args.generator_type not in generator_map:
            print(f"❌ Unknown generator type: {args.generator_type}")
            return
        
        generate_func = generator_map[args.generator_type]
        
        # Create output directory
        os.makedirs("outputs/batch", exist_ok=True)
        
        # Generate videos
        successful = 0
        for i, prompt in enumerate(args.prompts):
            print(f"\nGenerating video {i+1}/{len(args.prompts)}: {prompt[:50]}...")
            
            try:
                # Create args object for generation
                batch_args = argparse.Namespace()
                batch_args.prompt = prompt
                batch_args.output = f"outputs/batch/{args.generator_type}_{i:03d}.mp4"
                
                # Add common args
                batch_args.duration = args.duration
                batch_args.resolution = args.resolution
                batch_args.seed = args.seed + i if args.seed else None
                
                # Add generator-specific args
                if args.generator_type == "cinematic":
                    batch_args.style = args.style
                    batch_args.camera_movement = args.camera_movement
                    batch_args.lighting = args.lighting
                    batch_args.color_grading = args.color_grading
                elif args.generator_type == "product":
                    batch_args.image = args.image
                    batch_args.animation = args.animation
                    batch_args.style = args.style
                    batch_args.background = args.background
                    batch_args.lighting = args.lighting
                elif args.generator_type == "explainer":
                    batch_args.animation_style = args.animation_style
                    batch_args.educational_preset = args.educational_preset
                    batch_args.character_consistency = args.character_consistency
                    batch_args.education_level = args.education_level
                
                generate_func(batch_args)
                successful += 1
                
            except Exception as e:
                print(f"❌ Failed to generate video {i+1}: {e}")
        
        print(f"\n✅ Batch generation complete: {successful}/{len(args.prompts)} videos generated")
    
    def create_sequence(self, args):
        """Create a sequence of videos."""
        print("🎬 Creating cinematic sequence...")
        
        gen = self.get_generator("cinematic")
        
        if args.generator_type == "cinematic":
            sequence_path = gen.create_story_sequence(
                prompts=args.prompts,
                transitions=args.transitions,
                output_path=args.output or "outputs/cinematic_sequence.mp4"
            )
        elif args.generator_type == "product":
            if not args.images:
                print("❌ Product sequence requires --images parameter")
                return
            
            sequence_path = gen.generate_product_sequence(
                product_images=args.images,
                transitions=args.transitions,
                commercial_style=args.style,
                output_path=args.output or "outputs/product_sequence.mp4"
            )
        elif args.generator_type == "explainer":
            if not args.series_title:
                print("❌ Explainer series requires --series-title parameter")
                return
            
            video_paths = gen.create_educational_series(
                topics=args.prompts,
                series_title=args.series_title,
                animation_style=args.animation_style,
                educational_preset=args.educational_preset,
                output_dir="outputs/explainer_series"
            )
            sequence_path = video_paths[0] if video_paths else None
        else:
            print(f"❌ Unknown generator type for sequence: {args.generator_type}")
            return
        
        if sequence_path:
            print(f"✅ Sequence generated: {sequence_path}")
        else:
            print("❌ Failed to generate sequence")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="VideoGen CLI - Generate videos using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a cinematic video
  python cli.py cinematic "A lone astronaut walking on Mars at sunset" --style hollywood --camera dolly_in
  
  # Generate a product demo
  python cli.py product --image product.jpg --prompt "Sleek modern laptop" --animation turntable_360
  
  # Generate an explainer video
  python cli.py explainer "Explain how solar panels work" --style flat_2d --educational scientific
  
  # List available presets
  python cli.py list-presets --generator cinematic
  
  # Batch generate videos
  python cli.py batch "Scene 1" "Scene 2" "Scene 3" --generator cinematic --style hollywood
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Cinematic generator
    cinematic_parser = subparsers.add_parser('cinematic', help='Generate cinematic videos')
    cinematic_parser.add_argument('prompt', help='Scene description')
    cinematic_parser.add_argument('--style', choices=['hollywood', 'indie', 'arthouse'], default='hollywood')
    cinematic_parser.add_argument('--camera', '--camera-movement', dest='camera_movement', 
                                choices=list(PresetManager().get_presets("camera_movements").keys()), 
                                default='static')
    cinematic_parser.add_argument('--lighting', choices=list(PresetManager().get_presets("lighting").keys()), 
                                default='natural')
    cinematic_parser.add_argument('--color', '--color-grading', dest='color_grading',
                                choices=list(PresetManager().get_presets("color_grading").keys()), 
                                default='cinematic')
    cinematic_parser.add_argument('--duration', type=float, default=5.0, help='Duration in seconds')
    cinematic_parser.add_argument('--resolution', default='512x512', help='Video resolution (WxH)')
    cinematic_parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    cinematic_parser.add_argument('--output', '-o', help='Output video path')
    
    # Product generator
    product_parser = subparsers.add_parser('product', help='Generate product demos')
    product_parser.add_argument('--image', '-i', help='Product image path')
    product_parser.add_argument('--prompt', '-p', help='Product description')
    product_parser.add_argument('--animation', choices=list(PresetManager().get_presets("product_animations").keys()), 
                              default='turntable_360')
    product_parser.add_argument('--style', choices=['clean', 'luxury', 'tech', 'lifestyle', 'minimalist'], 
                              default='clean')
    product_parser.add_argument('--background', choices=list(PresetManager().get_presets("product_backgrounds").keys()), 
                              default='studio_white')
    product_parser.add_argument('--lighting', choices=['commercial', 'product_focused', 'luxury', 'warm_lifestyle'], 
                              default='commercial')
    product_parser.add_argument('--duration', type=float, default=4.0, help='Duration in seconds')
    product_parser.add_argument('--resolution', default='512x512', help='Video resolution (WxH)')
    product_parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    product_parser.add_argument('--output', '-o', help='Output video path')
    
    # Explainer generator
    explainer_parser = subparsers.add_parser('explainer', help='Generate explainer videos')
    explainer_parser.add_argument('prompt', help='Topic to explain')
    explainer_parser.add_argument('--style', '--animation-style', dest='animation_style',
                                choices=list(PresetManager().get_presets("animation_styles").keys()), 
                                default='flat_2d')
    explainer_parser.add_argument('--educational', '--educational-preset', dest='educational_preset',
                                choices=list(PresetManager().get_presets("explainer_styles").keys()), 
                                default='scientific')
    explainer_parser.add_argument('--character', '--character-consistency', dest='character_consistency',
                                action='store_true', default=True)
    explainer_parser.add_argument('--level', '--education-level', dest='education_level',
                                choices=['kids', 'general', 'professional', 'expert'], 
                                default='general')
    explainer_parser.add_argument('--duration', type=float, default=6.0, help='Duration in seconds')
    explainer_parser.add_argument('--resolution', default='512x512', help='Video resolution (WxH)')
    explainer_parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    explainer_parser.add_argument('--output', '-o', help='Output video path')
    
    # List presets
    list_parser = subparsers.add_parser('list-presets', help='List available presets')
    list_parser.add_argument('--generator', choices=['cinematic', 'product', 'explainer'], 
                           help='Filter by generator type')
    
    # Batch generation
    batch_parser = subparsers.add_parser('batch', help='Generate multiple videos')
    batch_parser.add_argument('prompts', nargs='+', help='List of prompts or descriptions')
    batch_parser.add_argument('--generator', '--generator-type', dest='generator_type',
                            choices=['cinematic', 'product', 'explainer'], required=True)
    batch_parser.add_argument('--duration', type=float, default=5.0, help='Duration for all videos')
    batch_parser.add_argument('--resolution', default='512x512', help='Resolution for all videos')
    batch_parser.add_argument('--seed', type=int, help='Starting seed (will increment)')
    
    # Add generator-specific batch options
    batch_cinematic = batch_parser.add_argument_group('Cinematic options')
    batch_cinematic.add_argument('--style', choices=['hollywood', 'indie', 'arthouse'], default='hollywood')
    batch_cinematic.add_argument('--camera', '--camera-movement', dest='camera_movement')
    batch_cinematic.add_argument('--lighting')
    batch_cinematic.add_argument('--color', '--color-grading', dest='color_grading')
    
    batch_product = batch_parser.add_argument_group('Product options')
    batch_product.add_argument('--image', help='Product image (optional)')
    batch_product.add_argument('--animation')
    batch_product.add_argument('--bg', '--background')
    batch_product.add_argument('--product-lighting')
    
    batch_explainer = batch_parser.add_argument_group('Explainer options')
    batch_explainer.add_argument('--animation-style')
    batch_explainer.add_argument('--educational-preset')
    batch_explainer.add_argument('--no-character', dest='character_consistency', action='store_false')
    batch_explainer.add_argument('--education-level')
    
    # Sequence generation
    sequence_parser = subparsers.add_parser('sequence', help='Create video sequences')
    sequence_parser.add_argument('prompts', nargs='+', help='List of prompts for sequence')
    sequence_parser.add_argument('--generator', '--generator-type', dest='generator_type',
                               choices=['cinematic', 'product', 'explainer'], required=True)
    sequence_parser.add_argument('--transitions', nargs='*', help='Transition types between scenes')
    sequence_parser.add_argument('--images', nargs='*', help='Images for product sequences')
    sequence_parser.add_argument('--series-title', help='Title for explainer series')
    sequence_parser.add_argument('--style', help='Style for sequence')
    sequence_parser.add_argument('--animation-style', help='Animation style for explainer series')
    sequence_parser.add_argument('--output', '-o', help='Output path')
    
    # System info
    system_parser = subparsers.add_parser('system', help='System information and diagnostics')
    system_parser.add_argument('--info', action='store_true', help='Show system information')
    system_parser.add_argument('--memory', action='store_true', help='Show memory usage')
    
    args = parser.parse_args()
    
    # Create CLI instance
    cli = VideoGenCLI()
    
    # Handle commands
    if args.command == 'cinematic':
        cli.generate_cinematic(args)
    
    elif args.command == 'product':
        cli.generate_product(args)
    
    elif args.command == 'explainer':
        cli.generate_explainer(args)
    
    elif args.command == 'list-presets':
        cli.list_presets(args.generator)
    
    elif args.command == 'batch':
        cli.batch_generate(args)
    
    elif args.command == 'sequence':
        cli.create_sequence(args)
    
    elif args.command == 'system':
        if args.info:
            cli.show_system_info()
        elif args.memory:
            cli.show_memory_info()
        else:
            cli.show_system_info()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()