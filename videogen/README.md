# VideoGen - Cloud-Optimized Video Generation System

A clean, efficient fork of VideoCrafter2 optimized for cloud GPU environments, designed for cinematic storytelling, product demos, and animated explainers.

## 🚀 Features

### Core Capabilities
- **Cinematic Storytelling**: Professional camera movements, lighting controls, genre presets
- **Product Demos**: Clean motion generation from product images with brand consistency
- **Animated Explainers**: Character consistency, style controls, and storyboard support
- **Cloud-Optimized**: Memory-efficient inference, GPU memory management
- **Batch Processing**: Generate multiple clips efficiently
- **Easy Customization**: LoRA integration, style presets

### Use Case Presets
- **Cinematic**: Camera movements (dolly, pan, tilt), lighting (golden hour, neon, moody)
- **Product**: Turntable animations, hero shots, detail reveals
- **Explainers**: Character consistency, flat colors, smooth transitions

## 📦 Installation

### Google Colab (Recommended)
```bash
# Upload the main notebook and run
!git clone https://github.com/your-username/videogen.git
%cd videogen
!pip install -r requirements.txt
```

### Local Development
```bash
git clone https://github.com/your-username/videogen.git
cd videogen
pip install -r requirements.txt
```

## 🎬 Quick Start

### 1. Cinematic Storytelling
```python
from videogen import CinematicGenerator

gen = CinematicGenerator()
video = gen.generate(
    prompt="A lone astronaut walking on Mars at sunset",
    style="cinematic",
    camera_movement="slow_dolly_in",
    duration=5
)
```

### 2. Product Demos
```python
from videogen import ProductGenerator

gen = ProductGenerator()
video = gen.generate(
    image_path="product_hero.png",
    animation="turntable_360",
    style="clean_commercial"
)
```

### 3. Animated Explainers
```python
from videogen import ExplainerGenerator

gen = ExplainerGenerator()
video = gen.generate(
    prompt="A happy character explaining renewable energy",
    style="flat_animation",
    character_consistency=True
)
```

## 🛠️ Architecture

```
videogen/
├── core/
│   ├── generators/          # Base generators for each use case
│   ├── models/             # Model architectures and checkpoints
│   ├── utils/              # Core utilities
│   └── presets/            # Style and movement presets
├── interfaces/
│   ├── gradio_ui.py        # Web interface
│   ├── cli.py             # Command line interface
│   └── api.py             # REST API
├── training/
│   ├── fine_tuning.py     # LoRA and DreamBooth training
│   ├── datasets.py        # Dataset handling
│   └── evaluation.py      # Model evaluation
└── examples/
    ├── notebooks/         # Usage examples
    └── scripts/           # Common workflows
```

## 🎯 Configuration

### Cloud GPU Optimization
- Memory-efficient inference
- Automatic batch size adjustment
- GPU memory monitoring
- Gradient checkpointing for training

### Model Configurations
- **Fast Mode**: T4 GPU, 256x256, 2-4 second clips
- **Standard Mode**: A100 GPU, 512x512, 4-6 second clips  
- **High Quality**: A100 GPU, 1024x1024, 6-8 second clips

## 🔧 Customization

### Adding New Presets
```python
# Add custom camera movements
PRESET_CAMERA_MOVEMENTS["my_custom_move"] = {
    "rotation": (0, 0, 15),  # degrees
    "translation": (0, 0, 0.1),  # meters
    "duration": 3.0  # seconds
}
```

### Fine-tuning Models
```python
from videogen.training import LoRATrainer

trainer = LoRATrainer()
trainer.train_lora(
    base_model="videogen/models/videogen_base",
    dataset="my_custom_data",
    output_dir="my_tuned_model"
)
```

## 📊 Performance

| GPU | Resolution | Duration | Speed | Quality |
|-----|------------|----------|-------|---------|
| T4  | 256x256    | 2-4s     | Fast  | Good    |
| A100 | 512x512    | 4-6s     | Medium| High    |
| A100 | 1024x1024  | 6-8s     | Slow  | Ultra   |

## 🐛 Troubleshooting

### Common Issues
- **CUDA Out of Memory**: Enable gradient checkpointing, reduce batch size
- **Slow Generation**: Use smaller resolution for initial tests
- **Quality Issues**: Increase CFG scale, use longer prompts

### Support
- Create issues in the GitHub repository
- Check the examples notebooks
- Join our Discord community

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.

---

Built with ❤️ for the creator community