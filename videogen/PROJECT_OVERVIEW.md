# VideoGen - Cloud-Optimized Video Generation System

## 🎬 Overview

VideoGen is a comprehensive AI video generation system optimized for cloud GPU environments. Built as a clean fork of VideoCrafter2, it provides specialized generators for cinematic storytelling, product demonstrations, and educational explainer videos.

## 🌟 Key Features

### Core Capabilities
- **🎭 Cinematic Generator**: Professional camera movements, lighting control, and color grading
- **📦 Product Generator**: Commercial-quality product demonstrations with turntable animations
- **📚 Explainer Generator**: Educational content with character consistency and flat animation styles
- **☁️ Cloud Optimized**: Memory-efficient inference for Colab, Kaggle, and other cloud platforms
- **🎨 Preset System**: Pre-configured styles and reusable configurations
- **🔄 Batch Processing**: Generate multiple videos efficiently
- **📊 Memory Management**: GPU memory optimization and monitoring

### Advanced Features
- **Web Interface**: Easy-to-use Gradio UI for all generator types
- **CLI Interface**: Command-line access for automation and scripting
- **Training Modules**: LoRA and DreamBooth fine-tuning capabilities
- **Evaluation System**: Quality metrics and model evaluation tools
- **Flexible Architecture**: Modular design for easy customization

## 📁 Project Structure

```
VideoGen/
├── README.md                           # Main documentation
├── requirements.txt                    # Dependencies
├── setup.py                           # Automated setup script
├── quick_start.py                     # Quick start guide
├── 
├── core/                              # Core generation modules
│   ├── __init__.py
│   ├── generators/                    # Base and specialized generators
│   │   ├── base_generator.py          # Abstract base generator
│   │   ├── cinematic_generator.py     # Cinematic storytelling
│   │   ├── product_generator.py       # Product demonstrations
│   │   └── explainer_generator.py     # Educational content
│   └── utils/                         # Core utilities
│       ├── config.py                  # Configuration management
│       ├── memory.py                  # GPU memory optimization
│       └── presets.py                 # Preset management
├── 
├── interfaces/                        # User interfaces
│   ├── __init__.py
│   ├── gradio_ui.py                   # Web interface
│   └── cli.py                        # Command-line interface
├── 
├── training/                         # Training and fine-tuning
│   ├── __init__.py
│   ├── fine_tuning.py                 # LoRA and DreamBooth training
│   ├── datasets.py                    # Dataset management
│   └── evaluation.py                  # Quality metrics and evaluation
├── 
├── examples/                         # Examples and demos
│   ├── demo.py                       # Comprehensive system demo
│   └── notebooks/
│       └── VideoGen_Complete_Setup.ipynb  # Colab setup notebook
└── 
└── outputs/                          # Generated video outputs
    ├── cinematic/                     # Cinematic videos
    ├── product/                       # Product demos
    ├── explainer/                     # Educational videos
    └── series/                        # Video series
```

## 🚀 Quick Start

### Method 1: Automated Setup
```bash
# Clone and setup
git clone <repository-url>
cd VideoGen
python setup.py

# Activate environment
source videogen_env/bin/activate  # Linux/macOS
# or
videogen_env\Scripts\activate     # Windows

# Quick start
python quick_start.py
```

### Method 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run examples
python examples/demo.py

# Launch web interface
python interfaces/gradio_ui.py

# Use CLI
python interfaces/cli.py --help
```

### Method 3: Google Colab
1. Upload `examples/notebooks/VideoGen_Complete_Setup.ipynb` to Colab
2. Run the setup cells
3. Follow the interactive examples

## 💡 Usage Examples

### Cinematic Video Generation
```python
from core import CinematicGenerator

with CinematicGenerator() as gen:
    video_path = gen.generate(
        prompt="A lone astronaut walking on Mars at sunset",
        style="hollywood",
        camera_movement="slow_dolly_in",
        lighting="golden_hour",
        color_grading="cinematic",
        duration=5.0
    )
```

### Product Demo Generation
```python
from core import ProductGenerator

with ProductGenerator() as gen:
    video_path = gen.generate(
        image_path="smartphone.jpg",  # or use prompt
        animation_type="turntable_360",
        commercial_style="tech",
        background="studio_white",
        duration=4.0
    )
```

### Explainer Video Generation
```python
from core import ExplainerGenerator

with ExplainerGenerator() as gen:
    video_path = gen.generate(
        prompt="Explain how solar panels work with simple illustrations",
        animation_style="flat_2d",
        educational_preset="scientific",
        character_consistency=True,
        duration=8.0
    )
```

### Command Line Usage
```bash
# Generate cinematic video
python interfaces/cli.py cinematic "A lone astronaut on Mars" --style hollywood

# Generate product demo  
python interfaces/cli.py product --image product.jpg --animation turntable_360

# Generate explainer video
python interfaces/cli.py explainer "How solar panels work" --style flat_2d
```

## 🎨 Preset System

### Available Preset Categories
- **Camera Movements**: static, dolly_in, orbital, tracking, crane_up
- **Lighting**: natural, golden_hour, dramatic, neon, moonlight
- **Color Grading**: cinematic, filmic, noir, vibrant
- **Product Animations**: turntable_360, hero_slide, detail_reveal, floating_rotate
- **Animation Styles**: flat_2d, cartoon, isometric, paper_craft, corporate
- **Educational Presets**: scientific, business, healthcare, technology, kids, finance

### Custom Presets
```python
# Create custom preset
preset_manager = PresetManager()
custom_preset = {
    "name": "My Custom Style",
    "params": {
        "camera_movement": "orbital",
        "lighting": "dramatic",
        "color_grading": "cinematic"
    },
    "tags": ["custom", "cinematic"]
}

preset_manager.create_preset("my_presets", "my_style", custom_preset)
```

## ⚙️ Cloud GPU Optimization

### Memory Management
- Automatic GPU memory optimization
- Memory usage monitoring
- Batch size adjustment based on available memory
- Memory cleanup between generations

### Environment Detection
- **Google Colab**: Optimized for Colab GPU/TPU environments
- **Kaggle**: Kaggle-specific optimizations
- **AWS/GCP/Azure**: Cloud platform optimizations
- **Local**: Fallback optimizations for local setups

### Performance Settings
- **Fast Mode**: 384x384 resolution, 15 inference steps
- **Balanced Mode**: 512x512 resolution, 20 inference steps  
- **Quality Mode**: 768x768 resolution, 25 inference steps
- **Ultra Mode**: 1024x1024 resolution, 30 inference steps

## 📊 System Requirements

### Minimum Requirements
- Python 3.8+
- 8GB RAM
- 2GB storage space
- Internet connection for initial setup

### Recommended for Best Performance
- NVIDIA GPU with 12GB+ VRAM (RTX 3080, 4090, A100, etc.)
- 32GB+ RAM
- 50GB+ storage space
- CUDA 11.8+

### Cloud Platform Support
- **Google Colab**: Free T4 GPU, Pro A100 GPU
- **Kaggle**: P100 GPU
- **AWS/GCP/Azure**: Various GPU instance types
- **RunPod**: Easy GPU cloud deployment

## 🔧 Customization

### Adding New Generators
```python
from core.generators.base_generator import BaseGenerator

class CustomGenerator(BaseGenerator):
    def generate(self, prompt, **kwargs):
        # Custom generation logic
        pass
```

### Extending Presets
```python
# Add new camera movement
preset_manager.create_preset("camera_movements", "my_custom_move", {
    "params": {
        "rotation": (0, 0, 15),
        "translation": (0, 0, 0.1)
    },
    "description": "Custom camera movement"
})
```

### Training Custom Models
```python
from training.fine_tuning import LoRATrainer

trainer = LoRATrainer()
results = trainer.train_lora(
    base_model="videogen/models/videogen_base",
    dataset="my_custom_data",
    rank=16,
    epochs=5
)
```

## 📈 Evaluation and Monitoring

### Quality Metrics
- **Technical**: FPS, resolution, bitrate, codec
- **Visual**: Sharpness, contrast, color consistency, temporal coherence
- **Text Alignment**: Semantic similarity, content relevance
- **Overall Score**: Weighted combination of all metrics

### Monitoring
```python
from core.utils.memory import MemoryManager
import torch

memory_manager = MemoryManager(torch.device('cuda'))
memory_info = memory_manager.get_memory_info()

print(f"GPU Memory: {memory_info['gpu_percent_used']:.1f}%")
```

## 🐛 Troubleshooting

### Common Issues

**GPU Out of Memory**
```python
# Reduce resolution and batch size
gen = CinematicGenerator()
gen.default_params.update({
    "resolution": (384, 384),
    "batch_size": 1
})
```

**Slow Generation**
- Use lower resolution for testing
- Enable fp16 optimization
- Use memory_efficient mode
- Check GPU utilization

**Quality Issues**
- Increase inference steps
- Use higher guidance scale
- Check prompt specificity
- Verify model compatibility

### Performance Optimization
```python
from core.utils.config import Config

config = Config()
config.apply_quality_preset("fast")  # For speed
config.apply_quality_preset("quality")  # For quality
```

## 🤝 Contributing

### Development Setup
```bash
git clone <repository-url>
cd VideoGen
pip install -r requirements.txt
pip install -e .  # Install in development mode

# Run tests
python -m pytest tests/

# Run linting
flake8 core/
black core/
```

### Adding Features
1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **VideoCrafter2**: Base architecture and model framework
- **Stable Video Diffusion**: Image-to-video generation inspiration
- **AnimateDiff**: Motion generation techniques
- **Hugging Face**: Model hosting and infrastructure
- **Gradio**: Web interface framework

## 📞 Support

- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Comprehensive guides and API reference
- **Community**: Discord/Forum for user support
- **Examples**: Extensive example gallery and tutorials

## 🗺️ Roadmap

### Version 1.1
- [ ] Real model integration
- [ ] Advanced camera controls
- [ ] Character consistency improvements
- [ ] Batch processing optimizations

### Version 1.2
- [ ] Training pipeline completion
- [ ] Custom model fine-tuning
- [ ] Advanced evaluation metrics
- [ ] Cloud deployment scripts

### Version 2.0
- [ ] Multi-modal video generation
- [ ] Real-time generation
- [ ] Advanced editing features
- [ ] Mobile app interface

---

**Built with ❤️ for the creator community**

*VideoGen - Making AI video generation accessible, efficient, and powerful for everyone.*