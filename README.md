# Prompt Refinement Candidate Generator

This repository provides a **multi-variant prompt refinement and image generation pipeline** built on **Stable Diffusion XL (SDXL)**, LoRA adapters, CLIP scoring, and aesthetic prediction.  
It generates multiple candidate images from input prompts, iteratively refining both **WHAT** (content) and **HOW** (style) descriptions using an LLM.  

---

## Features
- **Prompt Expansion & Refinement**  
  Expands terse prompts into rich narrative descriptions and converts them into SDXL-compatible tags.  
  Iteratively refines prompts based on CLIP similarity and aesthetic scoring.  

- **Multi-Variant Image Generation**  
  Runs prompt variations across models, LoRAs, seeds, and weights.  

- **Automated Scoring**  
  - **CLIP score**: semantic alignment with the user’s original prompt.  
  - **Aesthetic score**: quality of the rendered image.  

- **Captioning & Metadata**  
  Uses a vision-language model to caption generated images and stores candidate metadata in JSON.  

- **LoRA Integration**  
  Automatically applies LoRA adapters with customizable triggers and weights.  

---

## Installation

### 1. Clone repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate safetensors pillow
pip install llama-cpp-python colorlog
```

### 4. Download models, loras, etc

You need to have some idea of how SDXL works. Grab some checkpoints, or loras from various online sources.

### 5. Update the python script

There are a lot of things to tweak in this script. Set the correct paths to your checkpoint, LORAs, set LORA options, add prompts. You can change the examples the various models get of good "what" and "how" prompts.

---

## Hugging Face Token Setup

This project retrieves **CLIP** and **captioning models** from Hugging Face.  
You must set your HF token before running:

```bash
huggingface-cli login
# OR set environment variable
export HF_TOKEN=hf_your_token_here
```

If you don’t have a token yet, create one at:  
- https://huggingface.co/settings/tokens

---

## Usage

### Basic run
```bash
python generate-candidates.py
```

- Input prompts are defined in `USER_PROMPTS` inside the script.  
- Outputs are saved under `multi_variant_runs/<timestamp>/`.  
  - Candidate images (`*.png`)  
  - Metadata JSON (`*.json`)  

### Example prompts (WHAT)
- “Towering world-tree with roots piercing starlit sky, branches cradling fragments of glowing cities”  
- “Forgotten underwater temple, coral-encrusted statues of strange gods, shafts of light piercing the surface”  

### Example prompts (HOW)
- “Cinematic digital painting style, dramatic rim lighting, glowing highlights”  
- “Dreamlike surrealism, glowing desert horizon, radiant reflections”  

---

## Project Structure
- `generate-candidates.py` — Main pipeline for generating, refining, and scoring prompts.  
- `models/` — Expected directory for SDXL checkpoints and LoRA adapters.  
- `multi_variant_runs/` — Output directory (auto-created per run).  

---

## Notes
- Ensure you have enough **GPU VRAM** (recommended ≥ 24GB).  
- The script aggressively frees memory after each round to allow large model usage.  
- If models are not found locally, Hugging Face Hub will be used (hence the required HF token).  

---

## License
MIT License (adapt as needed).  
