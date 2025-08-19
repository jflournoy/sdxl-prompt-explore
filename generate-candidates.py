#!/usr/bin/env python3
import torch, gc, json, itertools, re
from math import ceil
from PIL import Image
from datetime import datetime
from pathlib import Path
from diffusers import (
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline
)
from transformers import (
    CLIPProcessor, 
    CLIPModel, 
    Blip2Processor, 
    Blip2ForConditionalGeneration, 
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

)
from llama_cpp import Llama
import logging, colorlog
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
import unicodedata
import shutil
import random
from typing import List, Dict
from transformers import CLIPTokenizer, CLIPTextModel
from sd_embed.embedding_funcs import get_weighted_text_embeddings_sdxl


HIGHLIGHT_LEVEL = 25
EXPAND_LEVEL   = 21
REFINE_LEVEL   = 22
CRITIQUE_LEVEL = 23
SDXL_LEVEL     = 24
logging.addLevelName(HIGHLIGHT_LEVEL, "HIGHLIGHT")
logging.addLevelName(EXPAND_LEVEL,   "EXPAND")
logging.addLevelName(REFINE_LEVEL,   "REFINE")
logging.addLevelName(CRITIQUE_LEVEL, "CRITIQUE")
logging.addLevelName(SDXL_LEVEL,     "SDXL")

# Attach methods to Logger
def _expand(self, message, *args, **kwargs):
    if self.isEnabledFor(EXPAND_LEVEL):
        self._log(EXPAND_LEVEL, message, args, **kwargs)

def _refine(self, message, *args, **kwargs):
    if self.isEnabledFor(REFINE_LEVEL):
        self._log(REFINE_LEVEL, message, args, **kwargs)

def _critique(self, message, *args, **kwargs):
    if self.isEnabledFor(CRITIQUE_LEVEL):
        self._log(CRITIQUE_LEVEL, message, args, **kwargs)

def _sdxl(self, message, *args, **kwargs):
    if self.isEnabledFor(SDXL_LEVEL):
        self._log(SDXL_LEVEL, message, args, **kwargs)

logging.Logger.expand   = _expand
logging.Logger.refine   = _refine
logging.Logger.critique = _critique
logging.Logger.sdxl     = _sdxl

handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        'EXPAND':   'light_yellow',
        'REFINE':   'green',
        'CRITIQUE': 'blue',
        'SDXL':     'purple',
        'DEBUG':    'white',
        'INFO':     'cyan',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'bold_red',
        'HIGHLIGHT':'bold_purple',   # ← your special color
    }
))

logger = colorlog.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger.propagate = False

# ── CONFIG ───────────────────────────────────────────────────────────────────────
MODEL_PATH       = ["models/checkpoints/auraRENDERXL_v30.safetensors", 
]

LORAS = [
    {"path": "models/lora/fr4z3tt4.safetensors", 
     "trigger": "fr4z3tt4, frazetta style, illustration",
     "weights": [0.3, 0.7]},
    {"path": "models/lora/Moebius (Jean Giraud) Style.safetensors", 
     "trigger": "Jean Giraud style, illustration",
     "weights": [1.0, 1.5]}
]

ALWAYS_ADD_LORAS = None
SEEDS            = [411]
USER_PROMPTS     = [
    "Titanic roots piercing stars, colossal tree anchoring galaxies",
    "Crystal citadel adrift in stormlit void, sails of plasma unfurled",
    "Ancient leviathan rising from molten sea, obsidian spires trembling"
]


NEGATIVE_PROMPT = (
    "lowres,"
    "poorly drawn hands, poorly drawn face, "
    "text, watermark, logo, signature, " 
    "airbrushed, blurry, worst quality, "
    "low quality, normal quality, low-res, "
    "polar low-res, monochrome, grayscale, zombie, NSFW"
)
NEG_GUIDANCE = 2.0  # how strongly to apply the negative prompt
#Create dictionary with indexes for prompts
user_prompts     = [
    {"index": i, "prompt": p} for i, p in enumerate(USER_PROMPTS)
]
LLAMA_PATH       = "models/llms/capybarahermes-2.5-mistral-7b.Q5_K_S.gguf"
LLAMA_HREPO      = "TheBloke/CapybaraHermes-2.5-Mistral-7B-GGUF"
LLAMA_HFNAME     = "*Q5_K_M.gguf"
LLM_GPU_LAYERS   = 32
N_ROUNDS         = 15
BATCH_SIZE       = 10
BEAM_WIDTH       = 2
HEIGHT, WIDTH    = 896, 1152
STEPS            = 30
REFINE_START     = 0.8
REFINE_STEPS     = 10
REFINE_STRENGTH  = .3
GUIDANCE         =7.5
SCORE_ALPHA      = 0.75
OUTPUT_ROOT_TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUT_ROOT      = Path(f"multi_variant_runs/{OUTPUT_ROOT_TIMESTAMP}")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

EXAMPLE_PROMPTS_WHAT = """\
Example WHAT prompts:
1. Towering world-tree with roots piercing starlit sky, branches cradling fragments of glowing cities
2. Red-haired mage with chin-length hair, holding a crystal orb, black robe embroidered with silver runes, arcane symbols swirling in background
3. Ancient stone portal half-buried in moss, ringed with glowing runes, mist spilling from within
4. Caravan of airships drifting over jagged mountain peaks, sails glowing faintly against twilight sky
5. Surreal crystalline desert with geometric dunes, lone traveler leaving a luminous trail of footprints
6. Observatory built into a cliffside, platinum domes gleaming, telescopes aimed toward aurora-filled skies
7. Forgotten underwater temple, coral-encrusted statues of strange gods, shafts of light piercing the surface
8. Gothic wanderer resting on a mossy stone, pale skin, black skirt, holding an old tome, surrounded by forest ruins
"""

EXAMPLE_PROMPTS_HOW = """\
Example HOW prompts:
1. Warm lantern glow, rustic clutter, dreamy farmhouse palette, golden fields under firefly-filled twilight
2. Cinematic digital painting style, dramatic rim lighting, glowing highlights, soft natural haze, mysterious and romantic
3. Hyperdetailed realism, sharp focus, fog curling between gravestones, delicate gradients, eerie yet beautiful mood
4. Sweeping fantasy illustration, HDR glow, aerial perspective, dramatic mountain backdrop, ethereal clouds
5. Dreamlike surrealism, glowing desert horizon, radiant reflections, minimalist yet otherworldly composition
6. Dark moody palette, deep contrasts, night scene with starfield and glowing runes, richly atmospheric tones
7. Hyperrealistic hand-drawn style, fine cross-hatching, strong contrast, vintage illuminated manuscript feel
8. Cinematic fantasy rendering, shallow depth, glowing torches in darkness, soft shadows, dreamlike framing
"""


# === Instruction templates ===
ENHANCE_THE = "vividness"
EXPAND_INSTR_WHAT = (
    "You are an SDXL prompt expander for the CONTENT of an image.\n"
    "You will receive a brief user phrase describing a scene or subject.\n"
    "- Write a concise description (2-4 sentences) that vividly describes WHAT is in the scene: characters, objects, actions, setting, and mood.\n"
    "- Use immersive, sensory-rich prose to bring the content to life; avoid tag lists, bullet points, or generic placeholders.\n"
    "- Emphasize narrative coherence by clarifying relationships and activities.\n"
    f"- Enahnce the {ENHANCE_THE} of the CONTENT of the scene.\n"
)

EXPAND_INSTR_HOW = (
    "You are an SDXL prompt expander for the VISUAL STYLE of an image.\n"
    "You will receive a brief user phrase describing a scene or subject.\n"
    "- Write a concise description (2-4 sentences) that vividly describes HOW the image appears: lighting, composition, color palette, texture, and atmosphere.\n"
    "- Use concrete, descriptive language referencing photographic or cinematic techniques (e.g., depth of field, rim lighting, perspective).\n"
    "- Focus on visual tone and stylistic details; avoid generic labels and ensure clarity of style.\n"
    f"- Enahnce the {ENHANCE_THE} of the VISUAL STYLE of the scene.\n"
)

# ── Prompt critique instructions ───────────────────────────────────────────────────────────
CRITIQUE_INSTR_WHAT = (
    "You are an SDXL prompt critic focused on the CONTENT (WHAT) of a tag-based prompt.\n"
    "You will be given:\n"
    "- The original user phrase.\n"
    "- The current WHAT prompt (tag-style).\n"
    "- An image caption describing what the image looks like.\n"
    "- The CLIP score (1-100) matching the original phrase - THIS IS THE PRIMARY METRIC you should focus on improving.\n"
    "- The CLIP score is directly proportional to how well the image matches the user's intent.\n"
    "- The strength of your critique should be proportional to the CLIP score.\n"
    "Based on these, list concise bullet-point suggestions to improve the WHAT prompt to better match the user's intended content.\n"
    "- Highlight missing or unclear content elements that would improve CLIP score.\n"
    "- Suggest clarifications to subject, action, or setting to better align with original intent.\n"
    "- Recommend stronger visual hooks or focal details that match user's description.\n"
    "- Focus primarily on improving semantic alignment to boost CLIP score.\n"
    "- Output only bullet points; no extra commentary.\n"
)

CRITIQUE_INSTR_HOW = (
    "You are an SDXL prompt critic focused on the VISUAL STYLE (HOW) of a tag-based prompt.\n"
    "You will be given:\n"
    "- The original user phrase.\n"
    "- The current HOW prompt (tag-style).\n"
    "- An image caption describing what the image looks like.\n"
    "- The aesthetic score (0-10) reflecting image quality - THIS IS THE PRIMARY METRIC you should focus on improving.\n"
    "- The aesthetic score is directly proportional to how visually appealing and technically proficient the image is.\n"
    "- The strength of your critique should be proportional to the aesthetic score.\n"
    "Based on these, list concise bullet-point suggestions to improve the HOW prompt to enhance aesthetic quality.\n"
    "- Point out weak or generic lighting, composition, or style cues that could be improved.\n"
    "- Recommend specific photographic or cinematic techniques to enhance visual appeal.\n"
    "- Suggest adjustments to depth of field, perspective, or color palette to boost aesthetic score.\n"
    "- Focus primarily on visual quality elements rather than semantic content.\n"
    "- Output only bullet points; no extra commentary.\n"
)

REFINE_INSTR_WHAT = (
    "You are an SDXL prompt refiner focused on CONTENT (WHAT) of a tag-style prompt.\n"
    "You will be given:\n"
    "- The current WHAT prompt (tag-style).\n"
    "- A critique as bullet points focused on improving CLIP score.\n"
    "Based on these, output a revised WHAT prompt (tag-style).\n"
    "- Incorporate critique to improve content alignment with user intent.\n"
    "- Reorder tags for importance and semantic relevance.\n"
    "- You MAY use numeric weights in parentheses, e.g., (detail:1.2).\n"
    "- Enhance narrative coherence and focal details to boost CLIP score.\n"
    f"- Enhance the {ENHANCE_THE} of the CONTENT while maintaining alignment with user intent.\n"
    "- Focus on making the prompt better match what the user asked for.\n"
    "- Output only the revised tags as a comma-separated list.\n"
)

REFINE_INSTR_HOW = (
    "You are an SDXL prompt refiner focused on VISUAL STYLE (HOW) of a tag-style prompt.\n"
    "You will be given:\n"
    "- The current HOW prompt (tag-style).\n"
    "- A critique as bullet points focused on improving aesthetic score.\n"
    "Based on these, output a revised HOW prompt (tag-style).\n"
    "- Incorporate critique to enhance visual appeal and aesthetic quality.\n"
    "- Reorder tags to prioritize techniques that improve aesthetic score.\n"
    "- You MAY use numeric weights in parentheses, e.g., (soft lighting:1.3).\n"
    "- Focus on lighting, composition, texture, and stylistic techniques.\n"
    f"- Enhance the {ENHANCE_THE} of the VISUAL STYLE to maximize aesthetic appeal.\n"
    "- Focus on making the image more visually appealing and high quality.\n"
    "- Output only the revised tags as a comma-separated list.\n"
)

SDXL_CONVERT_INSTR_WHAT = (
    "You are a prompt-to-tag converter for the WHAT of a Stable Diffusion XL image prompt.\n"
    "You will be given a Description string.\n"
    "- If the Description is already a comma-separated list of tags, return it unchanged.\n"
    "- Otherwise, convert the Description into dense booru-style tags: focus on subject matter, characters, actions, objects, and scene elements.\n"
)

SDXL_CONVERT_INSTR_HOW = (
    "You are a prompt-to-tag converter for the HOW of a Stable Diffusion XL image prompt.\n"
    "You will be given a Description string.\n"
    "- If the Description is already a comma-separated list of style tags, return it unchanged.\n"
    "- Otherwise, convert the Description into dense booru-style tags: focus on lighting, composition, rendering method, atmosphere, camera angle, and stylistic effects.\n"
)

SDXL_CONVERT_INSTR = (
    "You are a prompt-to-tag converter for a Stable Diffusion XL image prompt.\n"
    "You will be given a Description string, and examples of descriptions of WHAT is in an image and HOW the image looks. \n"
    "- If the Description is already a comma-separated list of style tags, return it unchanged.\n"
    "- Otherwise, convert the Description into dense booru-style tags.\n" 
    "- Maintain the richness of the description.\n"
    "- Use the examples to guide your conversion, but do not copy them directly.\n"
    "- Feel free create a long tag list and use descriptive phrases!\n"
)

EXPAND_TOKENS     = 256
CRITIQUE_TOKENS   = 512
REFINE_TOKENS     = 256
SDXL_TOKENS       = 512
MAX_PROMPT_TOKENS = 512

logger.debug("Config: MODEL_PATH=%s, LORAS=%s, …", MODEL_PATH, [l['path'] for l in LORAS])

# ── INITIALIZE LLM ONCE TO AVOID TOKENIZER WARNINGS ────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
device_cpu = torch.device("cpu")  

logger.debug("Loading AESCORE model")

aes_model, aes_preprocessor = convert_v2_5_from_siglip(
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

logger.debug("Loading caption model and CLIP model")

# blip_proc = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# blip2      = Blip2ForConditionalGeneration.from_pretrained(
#             "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
#             ).to("cpu")

# loads both the ViT encoder and GPT-2 decoder under the hood
cap_model_name        = "nlpconnect/vit-gpt2-image-captioning"
cap_model             = VisionEncoderDecoderModel.from_pretrained(cap_model_name)
cap_feature_extractor = ViTImageProcessor.from_pretrained(cap_model_name)
cap_tokenizer         = AutoTokenizer.from_pretrained(cap_model_name)

clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

cap_model.half()
clip_model.half()
aes_model = aes_model.to(torch.bfloat16)

cap_model.to(device)
clip_model.to(device)
aes_model.to(device)
aes_model.eval()

def clean_prompt_for_sdxl(p: str) -> str:
    # 1. Replace all escape-like whitespace characters with a single space
    p = re.sub(r"[\r\n\t\f\v]+", " ", p)

    # 2. Normalize and strip Unicode to ASCII
    p = unicodedata.normalize("NFKD", p)
    p = p.encode("ascii", "ignore").decode("ascii")

    # 3. Collapse multiple spaces to one
    p = re.sub(r"\s+", " ", p)

    return p.strip()


def setup_gpu_llm(verbose=False):
    # 1) Off-load vision models
    clip_model.to(device_cpu)
    aes_model.to(device_cpu)

    if LLAMA_HREPO is not None:
        llm = Llama.from_pretrained(
            repo_id=LLAMA_HREPO,
            filename=LLAMA_HFNAME,
            verbose=verbose,
            n_ctx=2048,
            n_gpu_layers=LLM_GPU_LAYERS
        )
    else:
        llm = Llama(
            model_path=LLAMA_PATH,
            n_ctx=2048,
            n_gpu_layers=LLM_GPU_LAYERS,
        )
    
    meta = getattr(llm, 'metadata', {}) or {}
    layers = (meta.get('n_layer') or
              meta.get('n_layers') or
              meta.get('llama.block_count'))

    # 2) Spin up one GPU-accelerated Llama
    logger.debug("Loaded LLM with %d/%d GPU layers", LLM_GPU_LAYERS, int(layers))

    return llm

def teardown_gpu_llm(gpu_llm):
    # 1) Close LLM & free its memory
    gpu_llm.close()
    del gpu_llm
    torch.cuda.empty_cache()
    gc.collect()
    logger.debug("LLM closed, GPU memory freed")

    # 2) Bring vision models back
    clip_model.to(device)
    aes_model.to(device)

def ensure_unpacked_model_dir(
    single_file_path: str,
    suffix: str = "-diffusers",
    **load_kwargs
) -> str:
    """
    Given a single-file checkpoint, unpack it into a sibling folder named
    "<checkpoint-stem><suffix>" (default suffix="-diffusers"). If that folder
    already has model_index.json, nothing happens.

    Args:
      single_file_path: path to your .safetensors/.ckpt
      suffix:           suffix to append to the file stem for the folder
      load_kwargs:      kwargs for from_single_file (e.g. torch_dtype, device_map, low_cpu_mem_usage, local_files_only, offload_folder)

    Returns:
      str(path_to_unpacked_dir)
    """
    logger.debug("Checking for unpacked model dir: %s", single_file_path)
    single_file = Path(single_file_path)
    parent_dir  = single_file.parent
    stem        = single_file.stem
    output_dir  = parent_dir / f"{stem}{suffix}"

    model_index = output_dir / "model_index.json"
    if not model_index.is_file():
        logger.debug("Unpacking model to %s", output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # 1) load your single-file SDXL checkpoint
        pipe = StableDiffusionXLPipeline.from_single_file(
            str(single_file),
            **load_kwargs
        )
        # 2) export it into Diffusers layout
        pipe.save_pretrained(str(output_dir))
        #delete pipe and save memory
        del pipe
        torch.cuda.empty_cache()
        gc.collect()
    
    logger.debug("Unpacked model dir: %s", output_dir)

    return str(output_dir)

# ── LLM HELPERS ─────────────────────────────────────────────────────────────────
def call_llm(llm, messages, max_tokens=64, temperature=0.7,
             repeat_penalty=1.15, top_p=0.95, top_k=40, seed=None):
    """
    Send a list of ChatML messages to the LLM and return the assistant's reply.
    """
    chatml = "".join(
        f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
    )
    chatml += "<|im_start|>assistant\n"

    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    resp = llm(
        chatml,
        max_tokens=max_tokens,
        temperature=temperature,
        repeat_penalty=repeat_penalty,
        top_p=top_p,
        top_k=top_k,
        stop=["<|im_end|>"],
        seed=seed,
    )
    return resp["choices"][0]["text"].strip()

def convert_to_sdxl_format(an_llm, description: str, version: str) -> str:
    """
    Convert a descriptive sentence to a Booru-style SDXL prompt.
    """
    if version == 'what':
        system_msg = SDXL_CONVERT_INSTR_WHAT
    elif version == 'how':
        system_msg = SDXL_CONVERT_INSTR_HOW
    else:
        system_msg = SDXL_CONVERT_INSTR
    user_msg = (f"Description: \"{description}\"\n"
                f"Example WHAT text: \"{EXAMPLE_PROMPTS_WHAT}\"\n"
                f"Example HOW text: \"{EXAMPLE_PROMPTS_HOW}\"\n"
                "SDXL-style prompt:")
    return call_llm(an_llm, [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg}
    ], max_tokens=SDXL_TOKENS, temperature=0.7, top_p=0.90)

def expand_prompt(an_llm, cur: str, version: str) -> str:
    """
    Expand a terse prompt into a detailed description, then convert to SDXL format.
    """
    # Natural-language expansion
    sys_msg = EXPAND_INSTR_WHAT if version == 'what' else EXPAND_INSTR_HOW
    user_msg = f"Expand this idea into a detailed description: \"{cur}\""
    description = call_llm(an_llm, [
        {"role": "system", "content": sys_msg},
        {"role": "user",   "content": user_msg}
    ], max_tokens=EXPAND_TOKENS, temperature=0.85, top_p=0.95, top_k=100)
    logger.expand(f"{description}")
    
    return description

def combine_how_what(an_llm, prompt_what: str, prompt_how: str) -> str:
    """
    Use the LLM to combine WHAT and HOW prompts into a single SDXL prompt string.
    """
    system_msg = (
        "You are an SDXL prompt combiner. "
        "Given a WHAT prompt (describing content) and a HOW prompt (describing visual style), "
        "combine them into a single, comma-separated SDXL prompt that captures both the content and the style. "
        "Do not lose any important details from either prompt. "
        "Maintain a richly detailed and concise prompt that fully captures the WHAT and HOW prompts' meaning and intent. "
        "Output only the combined SDXL prompt."
    )
    user_msg = (
        f"WHAT prompt: {prompt_what}\n"
        f"HOW prompt: {prompt_how}\n"
        "Combined SDXL prompt:"
    )
    combined = call_llm(an_llm, [
        {"role": "system", "content": system_msg},
        {"role": "user",   "content": user_msg}
    ], max_tokens=MAX_PROMPT_TOKENS, temperature=0.7, top_p=0.90)

    logger.debug("Combined WHAT and HOW prompts: %s", combined)
    # sdxl_combined = convert_to_sdxl_format(an_llm, combined, version = '')
    # logger.sdxl(f"SDXL combined prompt: {sdxl_combined}")

    return combined

def refine_prompt(an_llm, cur: str, caption: str, clip_score: float, aes_score: float, user_prompt: str, version: str) -> str:
    """
    Critique the current SDXL prompt and directly refine it into a new SDXL tag list.
    """
    sys_msg_critique = CRITIQUE_INSTR_WHAT if version == 'what' else CRITIQUE_INSTR_HOW
    sys_msg_refine   = REFINE_INSTR_WHAT if version == 'what' else REFINE_INSTR_HOW

    # Only include the relevant score for the critique
    if version == 'what':
        score_info = f"Current generation CLIP score: {clip_score:.2f}"
    else:
        score_info = f"Current generation aesthetic score: {aes_score:.2f}"

    critique = call_llm(an_llm, [
        {"role": "system", "content": sys_msg_critique},
        {"role": "user", "content": (
            f"Original user prompt: \"{user_prompt}\"\n"
            f"Current generation prompt: \"{cur}\"\n"
            f"Current generation caption: \"{caption}\"\n"
            f"{score_info}\n"
            "List ways to improve the prompt:"
        )}
    ], max_tokens=CRITIQUE_TOKENS, temperature=0.6)
    logger.critique(critique)

    refined_tags = call_llm(an_llm, [
        {"role": "system", "content": sys_msg_refine},
        {"role": "user",   "content": (
            f"Initial tags: {cur}\n"
            f"Critique: {critique}\n"
            "Refine SDXL tags list:"
        )}
    ], max_tokens=REFINE_TOKENS, temperature=0.7, top_p=0.90)
    logger.refine(refined_tags)

    return refined_tags

def inject_trigger(prompt: str, trigger: str) -> str:
    # use a word-boundary match (case-insensitive) to avoid partial hits
    if not trigger:
        return prompt
    pattern = rf"\b{re.escape(trigger)}\b"
    if re.search(pattern, prompt, flags=re.IGNORECASE):
        return prompt
    return f"{trigger}, {prompt}"

def enable_optimizations(pipelines):
    for p in pipelines:
        p.enable_xformers_memory_efficient_attention()
        p.enable_attention_slicing()
        p.enable_sequential_cpu_offload()
        p.enable_vae_slicing()

def offload_and_free_pipeline(pipe):
    """Move key components of a Diffusers pipeline to CPU and clear GPU memory."""
    if hasattr(pipe, "unet") and pipe.unet is not None:
        pipe.unet.to_empty(device="cpu")
        pipe.unet = None
    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae.to_empty(device="cpu")
        pipe.vae = None
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        pipe.text_encoder.to_empty(device="cpu")
        pipe.text_encoder = None
    if hasattr(pipe, "image_encoder") and pipe.image_encoder is not None:
        pipe.image_encoder.to_empty(device="cpu")
        pipe.image_encoder = None
    if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
        pipe.safety_checker.to("cpu")
        pipe.safety_checker = None
    if hasattr(pipe, "feature_extractor") and pipe.feature_extractor is not None:
        pipe.feature_extractor.to("cpu")
        pipe.feature_extractor = None
    if hasattr(pipe, "tokenizer"):
        pipe.tokenizer = None

    torch.cuda.empty_cache()
    gc.collect()

def run_variant(user_prompt, prompt_index, model, lora_i, lora_path, lora_var_trigger, weight, seed):
    from copy import deepcopy

    # ── 1) LOAD BASE SDXL PIPELINE ───────────────────────────────────────────────
    UNPACKED_DIR = ensure_unpacked_model_dir(
        model,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        offload_folder="offload",
        local_files_only=True
    )
    
    # ── 3) RAW → EXPAND → REFINEMENTS ───────────────────────────────────────────────
    n_refines    = N_ROUNDS
    total_rounds = n_refines + 2   # 0=raw, 1=expanded, 2..=refinements

    adapter_name = "none"
    if lora_path is not None:
        adapter_name = Path(lora_path).stem
    adapter_name = f"{lora_i:02d}-{adapter_name}"

    model_pattern = r"models/checkpoints/([^_/]+)(?:_[^/]*)?\.safetensors"
    model_name = re.search(model_pattern, model).group(1)

    tag = f"{OUTPUT_ROOT_TIMESTAMP}-p{prompt_index}-{model_name}-{adapter_name}-{weight}-{seed}"

    all_candidates = []
    beam = []

    for r in range(total_rounds):
        parent_prompts_what = [None] * BATCH_SIZE
        parent_prompts_how = [None] * BATCH_SIZE
        if r == 0:
            prompts_what = [user_prompt] * BATCH_SIZE
            prompts_how  = prompts_what.copy()
            logger.debug("ROUND %d: RAW PROMPT: %s", r, user_prompt)
            continue
        elif r == 1:
            gpu_llm = setup_gpu_llm()
            prompts_what = [expand_prompt(gpu_llm, p, 'what') for p in prompts_what]
            prompts_how  = [expand_prompt(gpu_llm, p, 'how') for p in prompts_how]
            teardown_gpu_llm(gpu_llm)
        else:
            # Alternating refinement rounds
            refine_what = (r % 2 == 1)  # Odd rounds refine WHAT, even rounds refine HOW
            if refine_what:
                logger.debug("ROUND %d: Refining WHAT prompts only", r)
            else:
                logger.debug("ROUND %d: Refining HOW prompts only", r)

            gpu_llm = setup_gpu_llm()
            prompt_pairs_what = []
            prompt_pairs_how = []
            per_parent = BATCH_SIZE // BEAM_WIDTH

            for entry in beam:
                for _ in range(per_parent):
                    if refine_what:
                        new_p_what = refine_prompt(
                            gpu_llm, 
                            entry["gen_prompt"], 
                            entry["caption"], 
                            entry["clip_score"], 
                            entry["aes_score"], 
                            user_prompt, 
                            'what')
                        new_p_how  = entry["prompt_how"]
                    else:
                        new_p_what = entry["prompt_what"]
                        new_p_how  = refine_prompt(
                            gpu_llm, 
                            entry["gen_prompt"], 
                            entry["caption"], 
                            entry["clip_score"], 
                            entry["aes_score"], 
                            user_prompt, 
                            'how')
                    prompt_pairs_what.append((new_p_what, entry["prompt_what"]))
                    prompt_pairs_how.append((new_p_how, entry["prompt_how"]))

            teardown_gpu_llm(gpu_llm)
            prompts_what = [p for p, _ in prompt_pairs_what]
            prompts_how  = [p for p, _ in prompt_pairs_how]
            parent_prompts_what = [p for _, p in prompt_pairs_what]
            parent_prompts_how  = [p for _, p in prompt_pairs_how]
        
        logger.debug("ROUND %d", r)
        
        # 1) Build a deduped dict of {lora_path: weight}
        all_loras = {}
        trigger = None
        if ALWAYS_ADD_LORAS:
            for l in ALWAYS_ADD_LORAS:
                all_loras[l["path"]] = l["weights"][0]
                trigger = ", ".join(filter(None, [lora_var_trigger, l["trigger"]]))
        else:
            trigger = lora_var_trigger
        if lora_path is not None:
            all_loras[lora_path] = weight

        # Show original prompts side by side
        for i, (p_what, p_how) in enumerate(zip(prompts_what, prompts_how)):
            logger.debug("Prompt[%d] WHAT: %s", i, repr(p_what))
            logger.debug("Prompt[%d]  HOW: %s", i, repr(p_how))

        # Inject triggers and clean
        gpu_llm = setup_gpu_llm()
        prompts_to_generate = [combine_how_what(gpu_llm, w, h) for w, h in zip(prompts_what, prompts_how)]
        teardown_gpu_llm(gpu_llm)
        prompts_to_generate = [clean_prompt_for_sdxl(inject_trigger(p, trigger)) for p in prompts_to_generate]

        for p in prompts_to_generate:
            logger.debug("Prompt to generate: %s", repr(p))

        logger.debug("Pipe: Loading Stable Diffusion XL pipeline: %s", UNPACKED_DIR)

        base_pipe = StableDiffusionXLPipeline.from_pretrained(
            UNPACKED_DIR, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            offload_folder="offload", local_files_only=True
        )

        logger.debug("Encoding prompts with text encoder")
        # base_pipe.text_encoder = base_pipe.text_encoder.to(device)

        base_pipe.text_encoder.eval()

        prompt_embeds = []
        prompt_neg_embeds = []
        pooled_prompt_embeds = []
        negative_pooled_prompt_embeds = []
        for p in prompts_to_generate:
            (
                pe,
                pne,
                ppe,
                nppe
            ) = get_weighted_text_embeddings_sdxl(
                base_pipe,
                prompt = p,
                neg_prompt = NEGATIVE_PROMPT
            )
            prompt_embeds.append(pe)
            prompt_neg_embeds.append(pne)
            pooled_prompt_embeds.append(ppe)
            negative_pooled_prompt_embeds.append(nppe)

        refine_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            UNPACKED_DIR, 
            torch_dtype=torch.float16, low_cpu_mem_usage=True,
            offload_folder="offload", local_files_only=True
        )

        base_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            base_pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",  # This specifies SDE variant
            solver_order=2,                   # This makes it "2M" (2nd-order multistep)
            use_karras_sigmas=True,           # This applies Karras sigma scheduling
            solver_type="midpoint"            # Optional: can be "midpoint" or "heun"
        )

        refine_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            refine_pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++", 
            solver_order=2,                  
            use_karras_sigmas=True,          
            solver_type="midpoint"            
        )

        logger.debug("Enabling optimizations")
        enable_optimizations([base_pipe, refine_pipe])

        logger.debug(
            "PIPELINE INIT: model_path=%s, dtype=%s, low_cpu_mem_usage=%s, device_map=%s, offload_folder=%s",
            UNPACKED_DIR, torch.float16, True, "auto", "offload"
        )

        # ── COMBINED LoRA LOADING & APPLICATION ──────────────────────────────────────────────

        # 2) Load each LoRA into base_pipe + refine_pipe (but catch the duplicate‐adapter error on refine)
        adapters = []
        weights_list = []

        for path, w in all_loras.items():
            adapter = Path(path).stem
            adapters.append(adapter)
            weights_list.append(w)

            # — load into base_pipe (UNet + text encoders)
            logger.debug("Loading LoRA into base_pipe → %s @ %.2f", path, w)
            base_pipe.load_lora_weights(path, adapter_name=adapter)

            # — load into refine_pipe: UNet will apply, but skip text‐encoder if already present
            logger.debug("Loading LoRA into refine_pipe → %s @ %.2f", path, w)
            try:
                refine_pipe.load_lora_weights(path, adapter_name=adapter)
            except ValueError as e:
                if f"Adapter with name {adapter} already exists" in str(e):
                    logger.debug("Adapter %s already in refine_pipe; skipping text encoder load", adapter)
                else:
                    raise

        # 3) Activate them all at once
        if adapters:
            base_pipe.set_adapters(adapters, weights_list)
            refine_pipe.set_adapters(adapters, weights_list)
            logger.debug("LoRA ADAPTERS SET: %s @ %s", adapters, weights_list)
        
        logger.debug(
            "ROUND %d/%d: seed=%d, H×W=%dx%d, steps=%d, guidance=%.2f",
            r+1, total_rounds, seed, HEIGHT, WIDTH, STEPS, GUIDANCE
        )
        
        if r == 0:
            gens = [
                torch.Generator(device=device).manual_seed(seed + i)
                for i in range(BATCH_SIZE)
            ]
        else:
            gens = [torch.Generator(device=device).manual_seed(seed)] * BATCH_SIZE
        
        logger.debug("Generating latents")

        first_pass_latents = []
        with torch.autocast(device_type=device, dtype=torch.float16):
            for pe, pne, ppe, nppe, g in zip(
                prompt_embeds,
                prompt_neg_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
                gens,
            ):
                logger.debug("Generator: %s", g.initial_seed())
                first_pass_latents.append(
                    base_pipe(
                        # denoising_end=REFINE_START,
                        prompt_embeds=pe,
                        negative_prompt_embeds=pne,
                        pooled_prompt_embeds=ppe,
                        negative_pooled_prompt_embeds=nppe,
                        height=HEIGHT,
                        width=WIDTH,
                        original_size=(HEIGHT, WIDTH),
                        target_size=(HEIGHT, WIDTH),
                        crops_coords_top_left=(0, 0),
                        num_inference_steps=STEPS,
                        guidance_scale=GUIDANCE,
                        generator=g,
                        negative_guidance_scale=NEG_GUIDANCE,
                        negative_original_size=(HEIGHT, WIDTH),
                        negative_target_size=(HEIGHT, WIDTH),
                        negative_crops_coords_top_left=(0, 0),
                        output_type="latent"
                    ).images[0]
                )
        
        #ACTUAL_REFINE_STEPS = ceil(REFINE_STEPS / REFINE_STRENGTH)
        batch_imgs = []
        with torch.autocast(device_type=device, dtype=torch.float16):
            for l, pe, pne, ppe, nppe in zip(
                first_pass_latents, 
                prompt_embeds,
                prompt_neg_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds
            ):
                batch_imgs.append(
                        refine_pipe(
                        # denoising_start=REFINE_START,
                        prompt_embeds=pe,
                        negative_prompt_embeds=pne,
                        pooled_prompt_embeds=ppe,
                        negative_pooled_prompt_embeds=nppe,
                        num_inference_steps=int(ceil(REFINE_STEPS // REFINE_STRENGTH)),
                        strength=REFINE_STRENGTH,
                        image=l,
                        original_size=(HEIGHT, WIDTH),
                        target_size=(HEIGHT, WIDTH),
                        crops_coords_top_left=(0, 0),
                        guidance_scale=GUIDANCE,
                        negative_guidance_scale=NEG_GUIDANCE,
                        negative_original_size=(HEIGHT, WIDTH),
                        negative_target_size=(HEIGHT, WIDTH),
                        negative_crops_coords_top_left=(0, 0),
                        return_dict=True
                    ).images[0]
                )
        
        round_results = []

        for i, (p_what,       p_how,       p,                   parent_what,         parent_how,         img) in enumerate(
            zip(prompts_what, prompts_how, prompts_to_generate, parent_prompts_what, parent_prompts_how, batch_imgs)
        ):
            filename = f"{tag}_r{r}_{i}_cand.png"
            candidate_path = OUTPUT_ROOT / filename
            img.save(candidate_path)
            logger.debug("SAVED CANDIDATE IMAGE: %s", candidate_path)

            cap_inputs = cap_feature_extractor(images=img, return_tensors="pt")
            cap_pixel_values = cap_inputs.pixel_values.to(device)

            # — GENERATE —
            cap_gen_kwargs = {
                "max_length": 150, 
                "min_length": 25,   
                "length_penalty": 1.0,
                "no_repeat_ngram_size": 1,
                "do_sample": True, 
                "top_p": 0.85, 
                "temperature": 0.70
            }
            cap_output_ids = cap_model.generate(cap_pixel_values, **cap_gen_kwargs)

            # — DECODE —
            caption = cap_tokenizer.batch_decode(cap_output_ids, skip_special_tokens=True)[0].strip()
            logger.debug("CAPTION: %s", caption)

            # CLIP score
            clip_in = clip_proc(images=img, text=user_prompt, return_tensors="pt", padding=True, truncation=True).to(device)
            i_emb   = clip_model.get_image_features(pixel_values=clip_in.pixel_values)
            t_emb   = clip_model.get_text_features(input_ids=clip_in.input_ids, attention_mask=clip_in.attention_mask)
            clip_score   = (i_emb / i_emb.norm() * t_emb / t_emb.norm()).sum().item() * 100
            logger.debug("CLIPSCORE [%d]: %.2f", i, clip_score)

            # img is a PIL.Image in RGB
            aes_inputs = aes_preprocessor(images=img, return_tensors="pt")
            aes_pixel_values = aes_inputs.pixel_values.to(device=device, dtype=torch.bfloat16)

            with torch.no_grad():
                logits = aes_model(aes_pixel_values).logits    # lives on GPU
                aes_score = logits.squeeze().float().cpu().item()
            
            logger.debug("AESCORE [%d]: %.2f", i, aes_score)
            alpha = SCORE_ALPHA
            score = float(alpha * clip_score / 100.0 + (1 - alpha) * aes_score / 10.0)

            logger.debug("SCORE [%d]: %.2f", i, score)

            round_results.append({
                "gen_prompt": p,
                "prompt_what": p_what,
                "prompt_how": p_how,
                "parent_prompt_what": parent_what,
                "parent_prompt_how": parent_how,
                "file": str(candidate_path),
                "clip_score": clip_score,
                "aes_score": aes_score,
                "score": score,
                "caption": caption,
                "round": r,
                "index": i
            })

        all_candidates.extend(round_results)
        beam = sorted(beam + round_results, key=lambda x: x["score"], reverse=True)[:BEAM_WIDTH]
        logger.debug("TOP CANDIDATES: %s", beam)
        #empty gpu memory, clean up base and refine pipelines
        for p in (base_pipe, refine_pipe):
            # 1) Move all components to CPU
            offload_and_free_pipeline(p)
            # 2) Delete the pipeline
            del p
        torch.cuda.empty_cache()
        gc.collect()
        shutil.rmtree("offload", ignore_errors=True)
        logger.debug("Pipeline freed")

    # write just the raw candidate metadata
    (OUTPUT_ROOT / f"{tag}.json").write_text(json.dumps({
        "user_prompt": user_prompt,
        "lora": adapter_name,
        "weight": weight,
        "seed": seed,
        "candidates": all_candidates
    }, indent=2))
    logger.debug("SAVED CANDIDATE METADATA: %s", OUTPUT_ROOT / f"{tag}.json")
    # free any residual cache
    gc.collect()
    torch.cuda.empty_cache()

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # ── BUILD FLAT TASK LIST ─────────────────────────────────────────────────────
    tasks = []
    for prompt, model, (lora_i, lora_entry), sd in itertools.product(user_prompts, MODEL_PATH, enumerate(LORAS), SEEDS):
        for wt in lora_entry["weights"]:
            tasks.append({
                "prompt":  prompt["prompt"],
                "index":   prompt["index"],
                "model":   model,
                "lora_i":  lora_i,
                "path":    lora_entry["path"],
                "trigger": lora_entry["trigger"],
                "weight":  wt,
                "seed":    sd
            })

    total = len(tasks)
    logger.info("Starting multi-variant run: %d variants to process", total)

    try:
        for idx, t in enumerate(tasks, start=1):
            # Top-level progress log
            lora_name = Path(t["path"] or "").stem or "none"
            logger.log(HIGHLIGHT_LEVEL,
                "🔥 [%d/%d] → prompt #%d, model %s, LoRA '%s' @%.2f, seed %d 🔥 ",
                idx, total, t["index"], model, lora_name, t["weight"], t["seed"]
            )

            logger.info("Testing LLM loading. Will error out if out of memory.")
            gpu_llm = setup_gpu_llm()
            teardown_gpu_llm(gpu_llm)

            run_variant(
                t["prompt"],
                t["index"],
                t["model"],
                t["lora_i"],
                t["path"],
                t["trigger"],
                t["weight"],
                t["seed"]
            )

            logger.info("[%d/%d] ← done", idx, total)

        logger.info("All variants completed successfully")
    except Exception:
        logger.exception("Run terminated with exception")
    finally:
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Shutdown complete")
