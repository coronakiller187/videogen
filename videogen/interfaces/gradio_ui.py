"""
Gradio Web Interface for VideoGen
Provides an easy-to-use web interface for all video generation types
"""

import os
import sys
import gradio as gr
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add core modules to path
sys.path.append(str(Path(__file__).parent.parent))

from core import (
    CinematicGenerator, 
    ProductGenerator, 
    ExplainerGenerator,
    PresetManager
)


class VideoGenUI:
    """
    Gradio-based web interface for VideoGen.
    Provides easy access to all generator types.
    """
    
    def __init__(self, server_name: str = "0.0.0.0", server_port: int = 7860):
        """
        Initialize the VideoGen UI.
        
        Args:
            server_name: Server host address
            server_port: Server port
        """
        self.server_name = server_name
        self.server_port = server_port
        
        # Initialize generators and preset manager
        self.cinematic_gen = None
        self.product_gen = None
        self.explainer_gen = None
        self.preset_manager = PresetManager()
        
        # Create output directory
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        print("🎨 VideoGen UI initialized")
    
    def _get_generator(self, generator_type: str):
        """Get or create generator instance."""
        if generator_type == "cinematic":
            if self.cinematic_gen is None:
                self.cinematic_gen = CinematicGenerator()
            return self.cinematic_gen
        
        elif generator_type == "product":
            if self.product_gen is None:
                self.product_gen = ProductGenerator()
            return self.product_gen
        
        elif generator_type == "explainer":
            if self.explainer_gen is None:
                self.explainer_gen = ExplainerGenerator()
            return self.explainer_gen
    
    def cinematic_generate(self, 
                          prompt: str,
                          style: str,
                          camera_movement: str,
                          lighting: str,
                          color_grading: str,
                          duration: float,
                          resolution: str,
                          seed: Optional[int],
                          progress: gr.Progress) -> tuple:
        """
        Generate cinematic video.
        
        Returns:
            Tuple of (video_path, status_message)
        """
        try:
            progress(0, desc="Initializing cinematic generator...")
            gen = self._get_generator("cinematic")
            
            # Parse resolution
            width, height = map(int, resolution.split('x'))
            
            progress(20, desc="Generating cinematic video...")
            
            # Generate video
            output_path = str(self.output_dir / f"cinematic_{hash(prompt) % 10000}.mp4")
            
            video_path = gen.generate(
                prompt=prompt,
                style=style,
                camera_movement=camera_movement,
                lighting=lighting,
                color_grading=color_grading,
                duration=duration,
                resolution=(width, height),
                seed=seed,
                output_path=output_path
            )
            
            progress(100, desc="Complete!")
            
            return video_path, f"✅ Cinematic video generated successfully!\nSaved to: {video_path}"
            
        except Exception as e:
            error_msg = f"❌ Error generating cinematic video: {str(e)}"
            print(error_msg)
            return None, error_msg
    
    def product_generate(self,
                        image_path: str,
                        prompt: str,
                        animation_type: str,
                        commercial_style: str,
                        background: str,
                        product_lighting: str,
                        duration: float,
                        resolution: str,
                        seed: Optional[int],
                        progress: gr.Progress) -> tuple:
        """
        Generate product demo video.
        
        Returns:
            Tuple of (video_path, status_message)
        """
        try:
            progress(0, desc="Initializing product generator...")
            gen = self._get_generator("product")
            
            # Parse resolution
            width, height = map(int, resolution.split('x'))
            
            progress(20, desc="Generating product demo...")
            
            # Generate video
            output_path = str(self.output_dir / f"product_{hash(str(image_path) + prompt) % 10000}.mp4")
            
            video_path = gen.generate(
                image_path=image_path if image_path else None,
                prompt=prompt if prompt else None,
                animation_type=animation_type,
                commercial_style=commercial_style,
                background=background,
                product_lighting=product_lighting,
                duration=duration,
                resolution=(width, height),
                seed=seed,
                output_path=output_path
            )
            
            progress(100, desc="Complete!")
            
            return video_path, f"✅ Product demo generated successfully!\nSaved to: {video_path}"
            
        except Exception as e:
            error_msg = f"❌ Error generating product demo: {str(e)}"
            print(error_msg)
            return None, error_msg
    
    def explainer_generate(self,
                          prompt: str,
                          animation_style: str,
                          educational_preset: str,
                          character_consistency: bool,
                          education_level: str,
                          duration: float,
                          resolution: str,
                          seed: Optional[int],
                          progress: gr.Progress) -> tuple:
        """
        Generate explainer video.
        
        Returns:
            Tuple of (video_path, status_message)
        """
        try:
            progress(0, desc="Initializing explainer generator...")
            gen = self._get_generator("explainer")
            
            # Parse resolution
            width, height = map(int, resolution.split('x'))
            
            progress(20, desc="Generating explainer video...")
            
            # Generate video
            output_path = str(self.output_dir / f"explainer_{hash(prompt) % 10000}.mp4")
            
            video_path = gen.generate(
                prompt=prompt,
                animation_style=animation_style,
                educational_preset=educational_preset,
                character_consistency=character_consistency,
                education_level=education_level,
                duration=duration,
                resolution=(width, height),
                seed=seed,
                output_path=output_path
            )
            
            progress(100, desc="Complete!")
            
            return video_path, f"✅ Explainer video generated successfully!\nSaved to: {video_path}"
            
        except Exception as e:
            error_msg = f"❌ Error generating explainer video: {str(e)}"
            print(error_msg)
            return None, error_msg
    
    def get_preset_options(self, generator_type: str) -> Dict[str, List[str]]:
        """Get available options for presets based on generator type."""
        presets = {}
        
        if generator_type == "cinematic":
            presets = {
                "styles": list(self.preset_manager.get_presets("color_grading").keys()),
                "camera_movements": list(self.preset_manager.get_presets("camera_movements").keys()),
                "lighting": list(self.preset_manager.get_presets("lighting").keys()),
                "color_grading": list(self.preset_manager.get_presets("color_grading").keys())
            }
        
        elif generator_type == "product":
            presets = {
                "animations": list(self.preset_manager.get_presets("product_animations").keys()),
                "backgrounds": list(self.preset_manager.get_presets("product_backgrounds").keys()),
                "styles": ["clean", "luxury", "tech", "lifestyle", "minimalist"],
                "lighting": ["commercial", "product_focused", "luxury", "warm_lifestyle"]
            }
        
        elif generator_type == "explainer":
            presets = {
                "animation_styles": list(self.preset_manager.get_presets("animation_styles").keys()),
                "educational_presets": list(self.preset_manager.get_presets("explainer_styles").keys()),
                "education_levels": ["kids", "general", "professional", "expert"]
            }
        
        return presets
    
    def create_interface(self) -> gr.Interface:
        """Create the main Gradio interface."""
        
        # Define CSS styling
        css = """
        .container { 
            max-width: 1200px; 
            margin: auto; 
            padding: 20px;
        }
        .title { 
            text-align: center; 
            color: #2E86AB; 
            font-size: 2.5em; 
            margin-bottom: 10px;
        }
        .subtitle { 
            text-align: center; 
            color: #666; 
            margin-bottom: 30px;
        }
        .tab-content { 
            padding: 20px; 
            border: 1px solid #ddd; 
            border-radius: 10px;
            margin: 10px 0;
        }
        """
        
        with gr.Blocks(css=css, title="VideoGen - AI Video Generation") as interface:
            
            # Header
            gr.HTML("""
            <div class="container">
                <h1 class="title">🎬 VideoGen</h1>
                <p class="subtitle">Cloud-Optimized AI Video Generation for Cinematic Storytelling, Product Demos, and Explainer Videos</p>
            </div>
            """)
            
            with gr.Tabs() as tabs:
                
                # Cinematic Generator Tab
                with gr.TabItem("🎭 Cinematic Generator"):
                    with gr.Column(elem_classes="tab-content"):
                        gr.HTML("<h2>🎬 Cinematic Storytelling</h2>")
                        
                        with gr.Row():
                            prompt = gr.Textbox(
                                label="Scene Description", 
                                placeholder="A lone astronaut walking on Mars at sunset, cinematic composition",
                                lines=3
                            )
                        
                        with gr.Row():
                            style = gr.Dropdown(
                                choices=["hollywood", "indie", "arthouse"],
                                value="hollywood",
                                label="Cinematic Style"
                            )
                            camera_movement = gr.Dropdown(
                                choices=list(self.preset_manager.get_presets("camera_movements").keys()),
                                value="static",
                                label="Camera Movement"
                            )
                        
                        with gr.Row():
                            lighting = gr.Dropdown(
                                choices=list(self.preset_manager.get_presets("lighting").keys()),
                                value="natural",
                                label="Lighting"
                            )
                            color_grading = gr.Dropdown(
                                choices=list(self.preset_manager.get_presets("color_grading").keys()),
                                value="cinematic",
                                label="Color Grading"
                            )
                        
                        with gr.Row():
                            duration = gr.Slider(2, 10, value=5, step=0.5, label="Duration (seconds)")
                            resolution = gr.Dropdown(
                                choices=["384x384", "512x512", "768x768", "1024x1024"],
                                value="512x512",
                                label="Resolution"
                            )
                        
                        with gr.Row():
                            seed = gr.Number(value=None, label="Seed (optional for reproducibility)")
                            generate_btn = gr.Button("🎬 Generate Cinematic Video", variant="primary")
                        
                        cinematic_output = gr.Video(label="Generated Video", interactive=False)
                        cinematic_status = gr.Textbox(label="Status", interactive=False)
                
                # Product Generator Tab
                with gr.TabItem("📦 Product Generator"):
                    with gr.Column(elem_classes="tab-content"):
                        gr.HTML("<h2>📦 Product Demonstrations</h2>")
                        
                        with gr.Row():
                            image_path = gr.Image(
                                label="Product Image (optional)", 
                                type="filepath"
                            )
                        
                        with gr.Row():
                            prompt = gr.Textbox(
                                label="Product Description", 
                                placeholder="Professional product demonstration of a sleek modern laptop",
                                lines=2
                            )
                        
                        with gr.Row():
                            animation_type = gr.Dropdown(
                                choices=list(self.preset_manager.get_presets("product_animations").keys()),
                                value="turntable_360",
                                label="Animation Type"
                            )
                            commercial_style = gr.Dropdown(
                                choices=["clean", "luxury", "tech", "lifestyle", "minimalist"],
                                value="clean",
                                label="Commercial Style"
                            )
                        
                        with gr.Row():
                            background = gr.Dropdown(
                                choices=list(self.preset_manager.get_presets("product_backgrounds").keys()),
                                value="studio_white",
                                label="Background"
                            )
                            product_lighting = gr.Dropdown(
                                choices=["commercial", "product_focused", "luxury", "warm_lifestyle"],
                                value="commercial",
                                label="Lighting"
                            )
                        
                        with gr.Row():
                            duration = gr.Slider(2, 8, value=4, step=0.5, label="Duration (seconds)")
                            resolution = gr.Dropdown(
                                choices=["384x384", "512x512", "768x768"],
                                value="512x512",
                                label="Resolution"
                            )
                        
                        with gr.Row():
                            seed = gr.Number(value=None, label="Seed (optional)")
                            generate_btn = gr.Button("📦 Generate Product Demo", variant="primary")
                        
                        product_output = gr.Video(label="Generated Video", interactive=False)
                        product_status = gr.Textbox(label="Status", interactive=False)
                
                # Explainer Generator Tab
                with gr.TabItem("📚 Explainer Generator"):
                    with gr.Column(elem_classes="tab-content"):
                        gr.HTML("<h2>📚 Educational Explainer</h2>")
                        
                        with gr.Row():
                            prompt = gr.Textbox(
                                label="Explanation Topic", 
                                placeholder="Explain how solar panels work with simple illustrations",
                                lines=3
                            )
                        
                        with gr.Row():
                            animation_style = gr.Dropdown(
                                choices=list(self.preset_manager.get_presets("animation_styles").keys()),
                                value="flat_2d",
                                label="Animation Style"
                            )
                            educational_preset = gr.Dropdown(
                                choices=list(self.preset_manager.get_presets("explainer_styles").keys()),
                                value="scientific",
                                label="Educational Domain"
                            )
                        
                        with gr.Row():
                            character_consistency = gr.Checkbox(
                                value=True, 
                                label="Maintain Character Consistency"
                            )
                            education_level = gr.Dropdown(
                                choices=["kids", "general", "professional", "expert"],
                                value="general",
                                label="Target Audience"
                            )
                        
                        with gr.Row():
                            duration = gr.Slider(3, 12, value=6, step=0.5, label="Duration (seconds)")
                            resolution = gr.Dropdown(
                                choices=["384x384", "512x512", "768x768"],
                                value="512x512",
                                label="Resolution"
                            )
                        
                        with gr.Row():
                            seed = gr.Number(value=None, label="Seed (optional)")
                            generate_btn = gr.Button("📚 Generate Explainer Video", variant="primary")
                        
                        explainer_output = gr.Video(label="Generated Video", interactive=False)
                        explainer_status = gr.Textbox(label="Status", interactive=False)
                
                # Presets and Settings Tab
                with gr.TabItem("⚙️ Settings & Presets"):
                    with gr.Column(elem_classes="tab-content"):
                        gr.HTML("<h2>⚙️ Configuration & Presets</h2>")
                        
                        preset_info = gr.JSON(
                            value=self.preset_manager.list_presets(),
                            label="Available Presets"
                        )
                        
                        with gr.Row():
                            quality_preset = gr.Radio(
                                choices=["fast", "balanced", "quality", "ultra"],
                                value="balanced",
                                label="Quality Preset"
                            )
                            apply_btn = gr.Button("Apply Quality Preset")
                        
                        status_output = gr.Textbox(label="System Status")
                
            # Connect events
            generate_btn.click(
                self.cinematic_generate,
                inputs=[
                    prompt, style, camera_movement, lighting, color_grading,
                    duration, resolution, seed
                ],
                outputs=[cinematic_output, cinematic_status]
            )
            
            # Apply quality preset
            apply_btn.click(
                self.apply_quality_preset,
                inputs=[quality_preset],
                outputs=[status_output]
            )
        
        return interface
    
    def apply_quality_preset(self, preset: str) -> str:
        """Apply quality preset settings."""
        # This would apply settings to the generators
        return f"Applied {preset} quality preset. This would optimize generation settings."
    
    def launch(self, share: bool = False, debug: bool = False):
        """Launch the Gradio interface."""
        interface = self.create_interface()
        
        print(f"🚀 Launching VideoGen UI at http://{self.server_name}:{self.server_port}")
        print("Press Ctrl+C to stop the server")
        
        try:
            interface.launch(
                server_name=self.server_name,
                server_port=self.server_port,
                share=share,
                debug=debug,
                show_error=True
            )
        except KeyboardInterrupt:
            print("\n🛑 VideoGen UI stopped")
        except Exception as e:
            print(f"❌ Error launching UI: {e}")


def main():
    """Main function to run the VideoGen UI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="VideoGen Web Interface")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Share interface publicly")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Create and launch UI
    ui = VideoGenUI(server_name=args.host, server_port=args.port)
    ui.launch(share=args.share, debug=args.debug)


if __name__ == "__main__":
    main()