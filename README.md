# VideoGen

[![CI](https://github.com/coronakiller187/videogen/workflows/CI/badge.svg?branch=main)](https://github.com/coronakiller187/videogen/actions)
[![License](https://img.shields.io/github/license/coronakiller187/videogen)](https://github.com/coronakiller187/videogen/blob/main/LICENSE)

VideoGen is an AI-powered video generation toolkit designed for cloud GPU environments. It provides cinematic, product, and explainer generation presets, a web (Gradio) interface, and a CLI for scripted use.

## Quick start

0. If you haven't cloned the repository yet, clone and run the setup:

   ```bash
   git clone https://github.com/coronakiller187/videogen.git
   cd videogen
   python setup.py
   ```

1. Activate the virtual environment:
   - Linux / macOS:
     ```bash
     source videogen_env/bin/activate
     ```
   - Windows (PowerShell):
     ```powershell
     .\videogen_env\Scripts\Activate.ps1
     ```

2. Install dependencies (if the setup did not install them automatically):
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the web interface (Gradio):
   ```bash
   python scripts/launch_gradio.py
   ```

4. Use the CLI:
   ```bash
   python scripts/videogen_cli.py --help
   ```

5. Run the quick start/examples:
   ```bash
   python quick_start.py
   python examples/demo.py
   ```

6. Colab:
   - If you prefer Colab, open `examples/notebooks/VideoGen_Complete_Setup.ipynb` or the generated `VideoGen_Colab_Setup.ipynb` in the repository root.

## What the setup script does

Running `python setup.py` will:
- Check Python version and (optionally) GPU availability
- Create a virtual environment `videogen_env`
- Install Python dependencies from `requirements.txt`
- Create output, models, data, and logs directories
- Create `scripts/launch_gradio.py` and `scripts/videogen_cli.py`
- Create a default `config.yaml`
- Copy/setup a Colab notebook if present

If any step fails, `setup.py` will print an error and exit unless you run it with `--force`.

## Project layout (important files)

- `setup.py` — automated setup script
- `quick_start.py` — quick demonstration and system checks
- `requirements.txt` — Python dependencies
- `scripts/launch_gradio.py` — launcher for the web UI
- `scripts/videogen_cli.py` — CLI entrypoint
- `examples/` — example scripts and notebooks
- `models/` — placeholder for downloaded model weights
- `outputs/` — default output directory for generated videos
- `config.yaml` — default configuration created by setup

## Tips

- Start with a lower resolution (e.g., 384x384) for faster testing and lower GPU memory usage.
- Use `fp16` / memory-efficient modes if your hardware supports it.
- Monitor GPU memory while experimenting and adjust `batch_size` and `inference_steps` as needed.
- Use presets for consistent results.

## Troubleshooting

- If you hit dependency or CUDA-related issues, ensure your system Python and drivers are up-to-date.
- On GPU systems, install a matching PyTorch build per your CUDA version (see https://pytorch.org/get-started/locally/).
- If `requirements.txt` is missing, run `pip install -r requirements.txt` after activating the virtualenv, or use `--skip-deps` with `setup.py` to continue without installing dependencies.

## Contributing

Contributions, bug reports, and improvements are welcome. Please open an issue or a pull request with a clear description of the change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
